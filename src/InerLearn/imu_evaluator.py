"""
imu_evaluator.py
================
Evaluates and visualizes IMU bias correction results.
"""

import numpy as np, pandas as pd, torch, time, logging, plotly.graph_objects as go

logger = logging.getLogger("imu_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class IMUEvaluator:
    def __init__(self, acc_meas, gyro_meas, a_true, t, env, scaler, estimators, window=40, dt=0.05):
        self.acc_meas=np.nan_to_num(acc_meas)
        self.gyro_meas=np.nan_to_num(gyro_meas)
        self.a_true=a_true; self.t=t; self.env=env
        self.scaler=scaler; self.estimators=estimators
        self.window=window; self.dt=dt
        self.results=None; self.err_curves={}; self.traj_corr={}

    @staticmethod
    def integrate(acc,dt):
        v=np.zeros_like(acc); p=np.zeros_like(acc)
        for i in range(1,len(acc)):
            v[i]=v[i-1]+acc[i]*dt
            p[i]=p[i-1]+v[i]*dt
        return p

    @staticmethod
    def rms(x): return float(np.sqrt(np.mean(x**2)))

    def corrected_traj(self,acc,gyro,model):
        n,win=len(acc),self.window
        X=np.hstack([acc,gyro])
        Xw=np.lib.stride_tricks.sliding_window_view(X,(win,X.shape[1])).reshape(-1,win*X.shape[1])
        Xs=self.scaler.transform(np.clip(Xw,-50,50))
        if hasattr(model,"predict"): bias=model.predict(Xs)
        else:
            with torch.no_grad():
                Xseq=torch.tensor(Xs.reshape(-1,win,6),dtype=torch.float32)
                bias=model(Xseq).numpy()
        bias=np.vstack([np.zeros((win-1,3)),bias])
        acc_corr=acc-bias
        return self.integrate(acc_corr,self.dt), bias

    def evaluate(self):
        start=time.perf_counter()
        p_true=self.integrate(self.a_true,self.dt)
        p_uncorr=self.integrate(self.acc_meas,self.dt)
        err_unc=np.linalg.norm(p_uncorr-p_true,axis=1)
        rms_unc=self.rms(err_unc); drift_unc=np.linalg.norm(p_uncorr[-1]-p_true[-1])
        rows=[]
        for n,m in self.estimators.items():
            try:
                p_corr,_=self.corrected_traj(self.acc_meas,self.gyro_meas,m)
                err_corr=np.linalg.norm(p_corr-p_true,axis=1)
                rms_corr=self.rms(err_corr)
                drift_corr=np.linalg.norm(p_corr[-1]-p_true[-1])
                self.err_curves[n]=err_corr; self.traj_corr[n]=p_corr
                rows.append([n,rms_unc,rms_corr,drift_unc,drift_corr])
                logger.info(f"{n:15s} RMS={rms_corr:.3f} DriftRed={(1-drift_corr/drift_unc)*100:.1f}%")
            except Exception as e: logger.error(f"❌ {n}: {e}")
        df=pd.DataFrame(rows,columns=["Model","RMS Uncorr","RMS Corr","Drift Uncorr","Drift Corr"])
        df["Drift Reduction (%)"]=100*(1-df["Drift Corr"]/df["Drift Uncorr"])
        self.results=df
        logger.info("\n"+df.to_string(index=False))
        logger.info(f"🏁 Evaluation took {time.perf_counter()-start:.2f}s")
        return df

    def plot_error_curves(self):
        if self.results is None: raise RuntimeError("Run evaluate() first")
        p_true=self.integrate(self.a_true,self.dt)
        p_unc=self.integrate(self.acc_meas,self.dt)
        err_unc=np.linalg.norm(p_unc-p_true,axis=1)
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=err_unc,x=self.t,name="Uncorrected",line=dict(color="black",dash="dash")))
        for n,e in self.err_curves.items():
            fig.add_trace(go.Scatter(y=e,x=self.t,name=f"{n} Corrected"))
        fig.update_layout(title="Position Error Over Time",xaxis_title="Time [s]",yaxis_title="Position Error [m]")
        fig.show()

    def plot_trajectories(self):
        if self.results is None: raise RuntimeError("Run evaluate() first")
        p_true=self.integrate(self.a_true,self.dt)
        p_unc=self.integrate(self.acc_meas,self.dt)
        fig=go.Figure()
        fig.add_trace(go.Scatter3d(x=p_true[:,0],y=p_true[:,1],z=p_true[:,2],mode='lines',line=dict(color='green',width=4),name='True'))
        fig.add_trace(go.Scatter3d(x=p_unc[:,0],y=p_unc[:,1],z=p_unc[:,2],mode='lines',line=dict(color='red',width=2,dash='dash'),name='Uncorrected'))
        for n,p in self.traj_corr.items():
            fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',name=f"{n} Corrected"))
        for (xo,yo) in getattr(self.env,"obstacles",[]):
            Xo=xo+1.5*np.cos(np.linspace(0,2*np.pi,20)); Yo=yo+1.5*np.sin(np.linspace(0,2*np.pi,20))
            fig.add_trace(go.Scatter3d(x=Xo,y=Yo,z=np.zeros(20),mode='lines',line=dict(color='red',width=4),name='Obstacle'))
        fig.update_layout(scene=dict(aspectmode='data'),title="3D Trajectories",height=700)
        fig.show()
