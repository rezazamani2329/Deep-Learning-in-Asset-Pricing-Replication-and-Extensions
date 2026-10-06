"""Full-schedule matched LSTM/Transformer GAN experiment (notebook 08 full)."""
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
    def forward(self, x): return self.lstm(x)[0]

class CausalTransformerEncoder(nn.Module):
    def __init__(self, output_dim, d_model=32, nhead=4, layers=2, dropout=.05):
        super().__init__()
        self.input_projection = nn.Linear(178, d_model)
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
        z = self.input_projection(x) + self.positions[:,:t]
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
    def forward(self, x, h): return self.output(torch.cat((x,h),dim=1))

class GAN(nn.Module):
    def __init__(self, kind):
        super().__init__()
        encoder = MacroLSTMEncoder if kind=='LSTM' else CausalTransformerEncoder
        self.sdf_macro_encoder = encoder(4)
        self.cond_macro_encoder = encoder(32)
        self.sdf_network = SDFWeightNetwork()
        self.conditional_network = ConditionalInstrumentNetwork()


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


def pricing_loss(model, panel, macro, conditional, frozen_raw=None, frozen_instruments=None):
    if frozen_raw is None:
        states=model.sdf_macro_encoder(macro)[0]
        raw,_,_=factor_and_scores(model,panel,states)
    else: raw=frozen_raw
    if conditional:
        if frozen_instruments is None:
            h=model.cond_macro_encoder(macro)[0]
            g=model.conditional_network(panel['x'],h[panel['month']])
        else: g=frozen_instruments
    else: g=torch.ones(len(panel['r']),1,device=macro.device)
    contributions=((1-raw[panel['month']])*panel['r']).unsqueeze(1)*g
    sums=torch.zeros(panel['N'],g.shape[1],device=macro.device).index_add(0,panel['slot'],contributions)
    active=panel['counts']>0
    moments=sums[active]/panel['counts'][active,None]
    return ((panel['counts'][active]/panel['T'])*moments.square().sum(1)).mean()


def train_member(kind, seed, panel, macro, config, output_dir, fingerprint):
    seed_all(seed)
    model=GAN(kind).to(macro.device)
    checkpoint=output_dir/f'{kind.lower()}_seed_{seed}.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=macro.device,weights_only=False)
        if saved['fingerprint']==fingerprint and saved['config']==config:
            model.load_state_dict(saved['state_dict']); model.eval()
            print(f'{kind} seed {seed}: reused matching completed checkpoint',flush=True)
            return model,pd.DataFrame(saved['history'])
        raise ValueError(f'Checkpoint configuration differs: {checkpoint}. Choose a new experiment directory.')
    sdf_params=list(model.sdf_macro_encoder.parameters())+list(model.sdf_network.parameters())
    cond_params=list(model.cond_macro_encoder.parameters())+list(model.conditional_network.parameters())
    history=[]; started=time.monotonic()
    for stage,epochs in enumerate(config['epochs'],1):
        for p in sdf_params:p.requires_grad_(stage!=2)
        for p in cond_params:p.requires_grad_(stage==2)
        model.sdf_macro_encoder.train(stage!=2); model.sdf_network.train(stage!=2)
        model.cond_macro_encoder.train(stage==2); model.conditional_network.train(stage==2)
        optimizer=torch.optim.Adam(cond_params if stage==2 else sdf_params,lr=config['learning_rate'])
        frozen_raw=frozen_g=None
        with torch.no_grad():
            if stage==2:
                h=model.sdf_macro_encoder(macro)[0]
                frozen_raw=factor_and_scores(model,panel,h)[0].detach()
            if stage==3:
                h=model.cond_macro_encoder(macro)[0]
                frozen_g=model.conditional_network(panel['x'],h[panel['month']]).detach()
        best_adversary_loss = -float('inf')
        best_adversary_state = None
        best_adversary_epoch = None
        for epoch in range(1,epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss=pricing_loss(model,panel,macro,stage!=1,frozen_raw,frozen_g)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {kind}/{seed}/{stage}/{epoch}')
            value=float(loss.detach().cpu())
            (-loss if stage==2 else loss).backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            selection_loss = None
            if stage == 2:
                # Evaluate the actual post-update state without dropout noise.
                model.cond_macro_encoder.eval(); model.conditional_network.eval()
                with torch.no_grad():
                    selection_loss = float(pricing_loss(model,panel,macro,True,frozen_raw).cpu())
                if not np.isfinite(selection_loss): raise RuntimeError('Nonfinite checkpoint score')
                if selection_loss > best_adversary_loss:
                    best_adversary_loss = selection_loss
                    best_adversary_epoch = epoch
                    best_adversary_state = {
                        'macro': copy.deepcopy(model.cond_macro_encoder.state_dict()),
                        'network': copy.deepcopy(model.conditional_network.state_dict())}
                model.cond_macro_encoder.train(); model.conditional_network.train()
            history.append(dict(architecture=kind,seed=seed,stage=stage,epoch=epoch,
                                loss=value,adversary_selection_loss=selection_loss))
            if epoch==1 or epoch%64==0 or epoch==epochs:
                print(f'{kind} seed={seed} stage={stage} epoch={epoch}/{epochs} loss={value:.8g} elapsed={time.monotonic()-started:.1f}s',flush=True)
        if stage == 2:
            model.cond_macro_encoder.load_state_dict(best_adversary_state['macro'])
            model.conditional_network.load_state_dict(best_adversary_state['network'])
            model.cond_macro_encoder.eval(); model.conditional_network.eval()
            with torch.no_grad():
                restored_loss = float(pricing_loss(model,panel,macro,True,frozen_raw).cpu())
            assert np.isclose(restored_loss,best_adversary_loss,rtol=1e-6,atol=1e-10)
            assert np.isclose(best_adversary_loss,max(row['adversary_selection_loss'] for row in history if row['stage']==2))
            print(f'{kind} seed={seed}: restored strongest adversary epoch {best_adversary_epoch}, training score={restored_loss:.8g}',flush=True)
        # Stage 3 uses its final epoch, as in the original fixed schedule.
        pd.DataFrame(history).to_csv(output_dir/f'{kind.lower()}_seed_{seed}_history.csv',index=False)
        del optimizer
    model.eval()
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
               config=config,fingerprint=fingerprint,history=history,seed=seed,architecture=kind),checkpoint)
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
    r=np.asarray(returns,dtype=float);vol=r.std(ddof=1)
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
