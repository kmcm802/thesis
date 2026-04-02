function result = boost_converter_hifi(duty_cycle, v_in, load_r, f_sw)
% boost_converter_hifi  CCM averaged state-space DC-DC boost converter model.
%
% INPUTS:
%   duty_cycle  Duty cycle D (dimensionless, typically 0.30-0.70)
%   v_in        Input voltage V_in (V, typically 10-20 V)
%   load_r      Load resistance R (Ohm, typically 5-50 Ohm)
%   f_sw        Switching frequency f_sw (Hz, typically 10e3-100e3 Hz)
%
% WHAT THIS FUNCTION COMPUTES:
%   Steady-state mean outputs via ode45 integration of averaged CCM state-space
%   model dx/dt = A*x + B, where x = [i_L, v_C]', with parasitics included.
%   Ripple outputs are computed from CCM analytical expressions at steady-state,
%   NOT extracted from simulated waveforms.
%
% WHAT THIS FUNCTION DOES NOT COMPUTE:
%   Switching transients, DCM operation, device non-linearities beyond fixed
%   forward voltage drop (V_d), thermal effects, or core saturation.
%
% CCM ASSUMPTION:
%   All operating points are assumed to be in continuous conduction mode.
%   DCM boundary is not verified for each sample.
%
% FIXED PARAMETERS (not function inputs):
%   L    = 100e-6 H   (inductor)
%   C    = 100e-6 F   (capacitor)
%   R_L  = 0.1 Ohm   (inductor series resistance)
%   R_sw = 0.05 Ohm  (switch on-resistance)
%   V_d  = 0.7 V     (diode forward voltage)
%
% ODE SOLVER: ode45, RelTol=1e-6, AbsTol=1e-9, MaxStep=T_sw/20
%
% OUTPUTS (fields of result struct):
%   result.v_out_mean   - ODE steady-state mean output voltage (V)
%   result.i_l_mean     - ODE steady-state mean inductor current (A)
%   result.i_l_ripple   - Analytical CCM: (V_in*D*T_sw)/L (A)
%   result.v_out_ripple - Analytical CCM: (I_out*D*T_sw)/C (V)
%   result.efficiency   - Power balance: P_out/P_in, capped at 0.99 (dimensionless)
L=100e-6; C=100e-6; R_L=0.1; R_sw=0.05; V_d=0.7;
D=duty_cycle; T_sw=1/f_sw;
A=[-(R_L+D*R_sw)/L, -(1-D)/L; (1-D)/C, -1/(load_r*C)];
B_vec=[(v_in-(1-D)*V_d)/L; 0];
tau=load_r*C; t_end=max(100*tau, 500*T_sw);
ode_fun=@(t,x) A*x+B_vec;
x0=[0;v_in];
opts=odeset('RelTol',1e-6,'AbsTol',1e-9,'MaxStep',T_sw/20);
[t,x]=ode45(ode_fun,[0,t_end],x0,opts);
ss=t>=(t_end-10*T_sw);
if sum(ss)<10; error('Not enough SS points'); end
result.v_out_mean=mean(x(ss,2));
result.i_l_mean=mean(x(ss,1));
i_out=result.v_out_mean/load_r;
result.i_l_ripple=(v_in*D*T_sw)/L;
result.v_out_ripple=(i_out*D*T_sw)/C;
P_out=result.v_out_mean^2/load_r;
P_in=v_in*result.i_l_mean;
if P_in<=0; result.efficiency=NaN;
else; result.efficiency=min(P_out/P_in,0.99); end
end
