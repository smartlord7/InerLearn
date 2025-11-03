# ==========================================
# imu_evaluator.py — Modular IMU Evaluation & Visualization Suite
# ==========================================

import numpy as np, pandas as pd, plotly.graph_objects as go, torch

class IMUEvaluator:
    def __init__(self, acc_meas, gyro_meas, a_true, t, env, scaler, estimators, window=40, dt=0.05):
        self.acc_meas = np.nan_to_num(acc_meas, nan=0.0, posinf=0.0, neginf=0.0)
        self.gyro_meas = np.nan_to_num(gyro_meas, nan=0.0, posinf=0.0, neginf=0.0)
        self.a_true = a_true
        self.t = t
        self.env = env
        self.scaler = scaler
        self.estimators = estimators
        self.window = window
        self.dt = dt
        self.results = None
        self.err_curves = {}

    # ---------- Utility ----------
    @staticmethod
    def integrate(acc, dt):
        """Integrate acceleration to position (dead reckoning)."""
        v = np.zeros_like(acc)
        p = np.zeros_like(acc)
        for i in range(1, len(acc)):
            v[i] = v[i-1] + acc[i]*dt
            p[i] = p[i-1] + v[i]*dt
        return p

    @staticmethod
    def rms(x):
        return float(np.sqrt(np.mean(x**2)))

    # ---------- Core correction ----------
    def corrected_traj(self, acc_meas, estimator):
        acc_corr = acc_meas.copy()
        preds = []
        for i in range(self.window, len(acc_meas)):
            window_data = np.hstack([acc_meas[i-self.window:i], self.gyro_meas[i-self.window:i]]).ravel()
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
            window_data = np.clip(window_data, -50, 50)
            window_scaled = self.scaler.transform(window_data.reshape(1, -1))
            # Predict bias
            if hasattr(estimator, "predict"):
                bias_pred = estimator.predict(window_scaled)[0]
            else:
                X_seq = torch.tensor(window_scaled.reshape(1, self.window, 6), dtype=torch.float32)
                with torch.no_grad():
                    bias_pred = estimator(X_seq).numpy()[0]
            preds.append(bias_pred)
            acc_corr[i] = acc_meas[i] - bias_pred
        return self.integrate(acc_corr, self.dt), np.array(preds)

    # ---------- Evaluation ----------
    def evaluate(self):
        p_true_ref = self.integrate(self.a_true, dt=self.dt)
        p_uncorr = self.integrate(self.acc_meas, dt=self.dt)
        err_uncorr = np.linalg.norm(p_uncorr - p_true_ref, axis=1)

        results = []
        for name, model in self.estimators.items():
            print(f"Evaluating {name} …")
            p_corr, _ = self.corrected_traj(self.acc_meas, model)
            err_corr = np.linalg.norm(p_corr - p_true_ref, axis=1)
            rms_unc = self.rms(err_uncorr)
            rms_corr = self.rms(err_corr)
            drift_unc = float(np.linalg.norm(p_uncorr[-1] - p_true_ref[-1]))
            drift_corr = float(np.linalg.norm(p_corr[-1] - p_true_ref[-1]))
            results.append([name, rms_unc, rms_corr, drift_unc, drift_corr])
            self.err_curves[name] = err_corr

        summary = pd.DataFrame(results, columns=["Model","RMS Uncorr","RMS Corr","Drift Uncorr","Drift Corr"])
        summary["Drift Reduction (%)"] = 100*(1 - summary["Drift Corr"]/summary["Drift Uncorr"])
        self.results = summary
        print("\n===== PERFORMANCE SUMMARY =====")
        print(summary.to_string(index=False))
        return summary

    # ---------- Visualization ----------
    def plot_error_curves(self):
        if self.results is None:
            raise RuntimeError("Run evaluate() before plotting.")
        err_uncorr = np.linalg.norm(
            self.integrate(self.acc_meas, self.dt) - self.integrate(self.a_true, self.dt),
            axis=1
        )
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            y=err_uncorr, x=self.t[:len(err_uncorr)],
            name="Uncorrected", line=dict(color="black", dash="dash")
        ))
        for name, err in self.err_curves.items():
            fig_err.add_trace(go.Scatter(y=err, x=self.t[:len(err)], name=f"{name} Corrected"))
        fig_err.update_layout(
            title="Position Error Over Time (IMU Bias Correction)",
            xaxis_title="Time [s]",
            yaxis_title="Position Error [m]",
            height=500
        )
        fig_err.show()

    def plot_trajectories(self):
        if self.results is None:
            raise RuntimeError("Run evaluate() before plotting.")
        p_true_ref = self.integrate(self.a_true, dt=self.dt)
        p_uncorr = self.integrate(self.acc_meas, dt=self.dt)
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=p_true_ref[:,0], y=p_true_ref[:,1], z=p_true_ref[:,2],
                                     mode='lines', line=dict(color='green', width=4), name='True'))
        fig3d.add_trace(go.Scatter3d(x=p_uncorr[:,0], y=p_uncorr[:,1], z=p_uncorr[:,2],
                                     mode='lines', line=dict(color='red', width=2, dash='dash'), name='Uncorrected'))
        for name, model in self.estimators.items():
            p_corr, _ = self.corrected_traj(self.acc_meas, model)
            fig3d.add_trace(go.Scatter3d(x=p_corr[:,0], y=p_corr[:,1], z=p_corr[:,2],
                                         mode='lines', name=f"{name} Corrected"))
        # Obstacles
        for (xo, yo) in self.env.obstacles:
            Xo = xo + 1.5*np.cos(np.linspace(0, 2*np.pi, 20))
            Yo = yo + 1.5*np.sin(np.linspace(0, 2*np.pi, 20))
            Zo = np.zeros(20)
            fig3d.add_trace(go.Scatter3d(x=Xo, y=Yo, z=Zo, mode='lines', line=dict(color='red', width=5), name='Obstacle'))
        fig3d.update_layout(scene=dict(aspectmode='data'),
                            title="3D Trajectories — True vs Corrected vs Uncorrected",
                            height=700)
        fig3d.show()

    def animate_best(self):
        if self.results is None:
            raise RuntimeError("Run evaluate() before animating.")
        best_model_name = self.results.sort_values("Drift Corr").iloc[0]["Model"]
        best_model = self.estimators[best_model_name]
        p_true_ref = self.integrate(self.a_true, self.dt)
        p_uncorr = self.integrate(self.acc_meas, self.dt)
        p_corr_best, _ = self.corrected_traj(self.acc_meas, best_model)

        frames=[]
        step = int(1/self.dt)
        for k in range(0, len(self.t), step):
            frames.append(go.Frame(data=[
                go.Scatter3d(x=p_true_ref[:k,0], y=p_true_ref[:k,1], z=p_true_ref[:k,2],
                             mode='lines', line=dict(color='green', width=4)),
                go.Scatter3d(x=p_uncorr[:k,0], y=p_uncorr[:k,1], z=p_uncorr[:k,2],
                             mode='lines', line=dict(color='red', width=2, dash='dash')),
                go.Scatter3d(x=p_corr_best[:k,0], y=p_corr_best[:k,1], z=p_corr_best[:k,2],
                             mode='lines', line=dict(color='blue', width=3))
            ], name=str(k)))

        fig_anim = go.Figure(
            data=[
                go.Scatter3d(x=[],y=[],z=[],mode='lines',line=dict(color='green',width=4),name='True'),
                go.Scatter3d(x=[],y=[],z=[],mode='lines',line=dict(color='red',width=2,dash='dash'),name='Uncorr'),
                go.Scatter3d(x=[],y=[],z=[],mode='lines',line=dict(color='blue',width=3),name=f'{best_model_name} Corr')
            ],
            layout=go.Layout(
                title=f"Real-time 3-D Playback ({best_model_name} Correction)",
                scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]', aspectmode='data'),
                updatemenus=[{
                    "buttons":[
                        {"args":[None,{"frame":{"duration":50,"redraw":True},"fromcurrent":True,"mode":"immediate"}],
                         "label":"▶ Play","method":"animate"},
                        {"args":[[None],{"frame":{"duration":0},"mode":"immediate"}],
                         "label":"⏸ Pause","method":"animate"}],
                    "direction":"left","pad":{"r":10,"t":87},"showactive":False,"type":"buttons",
                    "x":0.1,"xanchor":"right","y":0,"yanchor":"top"}]
            ),
            frames=frames
        )
        fig_anim.show()
