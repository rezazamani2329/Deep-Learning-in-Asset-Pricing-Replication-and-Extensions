"""PyTorch port of verified author GAN conventions, with a causal Transformer extension."""
from pathlib import Path
import copy, hashlib, json, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn

SPLITS = {'train': (0, 240), 'validation': (240, 300), 'test': (300, 600)}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class MacroLSTMEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__(); self.lstm = nn.LSTM(178, output_dim, batch_first=True)
        self.input_dropout = nn.Dropout(.05)
        # TensorFlow LSTMCell: one combined Glorot kernel, zero bias, forget bias +1.
        kernel = torch.empty(178+output_dim,4*output_dim)
        nn.init.xavier_uniform_(kernel)
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(kernel[:178].T)
            self.lstm.weight_hh_l0.copy_(kernel[178:].T)
            self.lstm.bias_ih_l0.zero_(); self.lstm.bias_hh_l0.zero_()
            self.lstm.bias_ih_l0[output_dim:2*output_dim] = 1
        self.lstm.bias_hh_l0.requires_grad_(False)
    def forward(self, x): return self.lstm(self.input_dropout(x))[0]

class CausalTransformerEncoder(nn.Module):
    def __init__(self, output_dim, d_model=32, nhead=4, layers=2, dropout=.05):
        super().__init__()
        self.input_projection = nn.Linear(178, d_model)
        self.input_dropout = nn.Dropout(.05)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                    dropout=dropout, activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output_projection = nn.Linear(d_model, output_dim)
        positions = torch.arange(600).float().unsqueeze(1)
        scales = torch.exp(torch.arange(0, d_model, 2).float()*(-np.log(10000.)/d_model))
        pe = torch.zeros(600, d_model)
        pe[:,0::2] = torch.sin(positions*scales); pe[:,1::2] = torch.cos(positions*scales)
        self.register_buffer('positions', pe.unsqueeze(0))
    def forward(self, x):
        t = x.shape[1]
        z = self.input_projection(self.input_dropout(x)) + self.positions[:,:t]
        # Boolean True entries prohibit attending to later observations.
        mask = torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        return torch.tanh(self.output_projection(self.encoder(z, mask=mask)))

class SDFWeightNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(50,64),nn.ReLU(),nn.Dropout(.05),
                     nn.Linear(64,64),nn.ReLU(),nn.Dropout(.05),nn.Linear(64,1))
    def forward(self, x, h): return self.network(torch.cat((x,h),dim=1)).squeeze(-1)

class ConditionalInstrumentNetwork(nn.Module):
    def __init__(self):
        super().__init__(); self.output = nn.Linear(78,8)
    def forward(self, x, h): return torch.tanh(self.output(torch.cat((x,h),dim=1)))

class GAN(nn.Module):
    def __init__(self, kind):
        super().__init__()
        encoder = MacroLSTMEncoder if kind=='LSTM' else CausalTransformerEncoder
        self.sdf_macro_encoder = encoder(4)
        self.cond_macro_encoder = encoder(32)
        self.sdf_network = SDFWeightNetwork()
        self.conditional_network = ConditionalInstrumentNetwork()
        for net in [self.sdf_network,self.conditional_network]:
            for layer in net.modules():
                if isinstance(layer,nn.Linear):
                    nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)


def load_panel(path, expected_dates, device):
    with np.load(path,allow_pickle=True) as a:
        data = a['data'].astype(np.float32)
        raw_dates = a['date'].copy()
        names = [v.decode() if isinstance(v,bytes) else str(v) for v in a['variable']]
    dates = pd.to_datetime([str(int(d))[:6] for d in raw_dates],format='%Y%m')
    assert pd.DatetimeIndex(dates).equals(expected_dates), path
    assert len(names)==47 and names[0].lower()=='ret', names
    mask = data[:,:,0] != np.float32(-99.99)
    month, slot = np.nonzero(mask)
    active_data = data[month,slot]
    assert np.isfinite(active_data).all()
    assert np.abs(active_data[:,1:]).max() <= .50001
    assert (mask.sum(1)>0).all()
    panel = dict(x=torch.tensor(active_data[:,1:],device=device),
        r=torch.tensor(active_data[:,0],device=device),
        month=torch.tensor(month,device=device),slot=torch.tensor(slot,device=device),
        counts=torch.tensor(mask.sum(0),dtype=torch.float32,device=device),
        T=data.shape[0], N=data.shape[1], dates=dates, names=names[1:])
    return panel


def factor_and_scores(model, panel, states, chunk_size=131072):
    scores = []
    for start in range(0,len(panel['r']),chunk_size):
        end=start+chunk_size
        scores.append(model.sdf_network(panel['x'][start:end],states[panel['month'][start:end]]))
    w=torch.cat(scores)
    raw=torch.zeros(panel['T'],device=w.device).index_add(0,panel['month'],w*panel['r'])
    gross=torch.zeros_like(raw).index_add(0,panel['month'],w.abs())
    return raw,gross,w


def pricing_loss(model, panel, macro, conditional, frozen_raw=None, frozen_instruments=None, macro_offset=0):
    if frozen_raw is None:
        states=model.sdf_macro_encoder(macro)[0,macro_offset:macro_offset+panel["T"]]
        raw,_,_=factor_and_scores(model,panel,states)
    else: raw=frozen_raw
    if conditional:
        if frozen_instruments is None:
            h=model.cond_macro_encoder(macro)[0,macro_offset:macro_offset+panel["T"]]
            g=model.conditional_network(panel['x'],h[panel['month']])
        else: g=frozen_instruments
    else: g=torch.ones(len(panel['r']),1,device=macro.device)
    contributions=((1-raw[panel['month']])*panel['r']).unsqueeze(1)*g
    sums=torch.zeros(panel['N'],g.shape[1],device=macro.device).index_add(0,panel['slot'],contributions)
    active=panel['counts']>0
    moments=sums[active]/panel['counts'][active,None]
    return ((panel['counts'][active]/panel['counts'].max())[:,None]*moments.square()).mean()


def _snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def validation_metrics(model, panels, macro):
    model.eval()
    prefix=macro[:,:300]
    loss=float(pricing_loss(model,panels['validation'],prefix,False,macro_offset=240).cpu())
    states=model.sdf_macro_encoder(prefix)[0,240:300]
    raw,_,_=factor_and_scores(model,panels['validation'],states)
    r=raw.cpu().numpy().astype(float)
    return loss,float(r.mean()/r.std(ddof=0))


def train_member(kind, seed, panel, macro, config, output_dir, fingerprint, validation_panels=None, full_macro=None):
    assert validation_panels is not None and full_macro is not None
    seed_all(seed)
    model=GAN(kind).to(macro.device)
    checkpoint=output_dir/f'{kind.lower()}_seed_{seed}.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=macro.device,weights_only=False)
        if saved['fingerprint']!=fingerprint or saved['config']!=config:
            raise ValueError(f'Checkpoint configuration differs: {checkpoint}')
        model.load_state_dict(saved['state_dict']);model.eval()
        print(f'{kind} seed {seed}: reused completed paper-convention checkpoint',flush=True)
        return model,pd.DataFrame(saved['history'])
    sdf_params=[p for m in [model.sdf_macro_encoder,model.sdf_network] for p in m.parameters() if p.requires_grad]
    cond_params=[p for m in [model.cond_macro_encoder,model.conditional_network] for p in m.parameters() if p.requires_grad]
    history=[];started=time.monotonic();selection={}
    for stage,epochs in enumerate(config['epochs'],1):
        for p in sdf_params:p.requires_grad_(stage!=2)
        for p in cond_params:p.requires_grad_(stage==2)
        # The authors' shared dropout placeholder is active for both arms in training.
        model.train()
        optimizer=torch.optim.Adam(cond_params if stage==2 else sdf_params,lr=config['learning_rate'],eps=1e-8)
        best=float('inf') if stage==1 else -float('inf')
        best_state=None;best_epoch=None
        if stage==2:
            # Author iterator repeats the full training data four times. Each pass
            # has a 64-step adversary search; its strongest checkpoint overwrites
            # the previous pass, and the final pass's winner enters Stage 3.
            for sub in range(config['sub_epoch']):
                best=-float('inf');best_state=None
                for epoch in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    loss=pricing_loss(model,panel,macro,True)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite adversarial loss')
                    value=float(loss.detach().cpu());(-loss).backward();optimizer.step()
                    if value>best:
                        best=value;best_state=_snapshot(model);best_epoch=epoch
                    history.append(dict(architecture=kind,seed=seed,stage=stage,epoch=epoch,
                                        sub_epoch=sub,loss=value,validation_loss=None,validation_raw_sharpe=None))
                print(f'{kind} seed={seed} stage=2 pass={sub+1}/4 strongest training score={best:.8g}',flush=True)
        else:
            for epoch in range(epochs):
                for sub in range(config['sub_epoch']):
                    optimizer.zero_grad(set_to_none=True)
                    loss=pricing_loss(model,panel,macro,stage==3)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite pricing loss')
                    loss.backward();optimizer.step()
                valid_loss,valid_sr=validation_metrics(model,validation_panels,full_macro)
                model.train()
                value=float(loss.detach().cpu())
                score=valid_loss if stage==1 else valid_sr
                if epoch>config['ignore_epoch'] and ((score<best) if stage==1 else (score>best)):
                    best=score;best_epoch=epoch;best_state=_snapshot(model)
                history.append(dict(architecture=kind,seed=seed,stage=stage,epoch=epoch,
                                    sub_epoch=config['sub_epoch'],loss=value,
                                    validation_loss=valid_loss,validation_raw_sharpe=valid_sr))
                if epoch==0 or (epoch+1)%64==0 or epoch+1==epochs:
                    print(f'{kind} seed={seed} stage={stage} epoch={epoch+1}/{epochs} val_loss={valid_loss:.7g} val_raw_SR={valid_sr:.4f} elapsed={time.monotonic()-started:.0f}s',flush=True)
        assert best_state is not None
        model.load_state_dict(best_state)
        selection[str(stage)]={'epoch_zero_based':best_epoch,'criterion':best}
        pd.DataFrame(history).to_csv(output_dir/f'{kind.lower()}_seed_{seed}_history.csv',index=False)
        print(f'{kind} seed={seed}: restored Stage {stage} selected checkpoint at epoch {best_epoch}',flush=True)
        del optimizer
    model.eval()
    torch.save(dict(state_dict=_snapshot(model),config=config,fingerprint=fingerprint,history=history,
                   seed=seed,architecture=kind,selection=selection),checkpoint)
    return model,pd.DataFrame(history)

@torch.no_grad()
def evaluate(model, panels, macro):
    model.eval(); states=model.sdf_macro_encoder(macro)[0]
    factors=[]
    for split,(start,end) in SPLITS.items():
        raw,gross,_=factor_and_scores(model,panels[split],states[start:end])
        if (gross<=1e-12).any():raise ValueError('Zero gross portfolio exposure')
        factors.append((raw/gross).cpu().numpy())
    return np.concatenate(factors)


def stats(returns):
    r=np.asarray(returns,dtype=float);vol=r.std(ddof=0)
    return dict(months=len(r),mean_monthly=r.mean(),volatility_monthly=vol,
        monthly_sharpe=r.mean()/vol,annualized_sharpe=np.sqrt(12)*r.mean()/vol,
        minimum_month=r.min(),maximum_month=r.max(),positive_month_fraction=(r>0).mean())


def fingerprint_files(paths):
    result={}
    for path in paths:
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        result[str(Path(path).resolve())]=h.hexdigest()
    return result

@torch.no_grad()
def raw_scores(model,panels,macro):
    model.eval();states=model.sdf_macro_encoder(macro)[0]
    return {split:factor_and_scores(model,panels[split],states[a:b])[2].cpu().numpy()
            for split,(a,b) in SPLITS.items()}

def ensemble_from_raw_scores(score_sums,panels,n_members):
    factors=[]
    for split in SPLITS:
        panel=panels[split]
        w=score_sums[split]/n_members
        month=panel['month'].cpu().numpy();r=panel['r'].cpu().numpy()
        raw=np.bincount(month,weights=w*r,minlength=panel['T'])
        gross=np.bincount(month,weights=np.abs(w),minlength=panel['T'])
        assert (gross>1e-12).all()
        factors.append(raw/gross)
    return np.concatenate(factors)
