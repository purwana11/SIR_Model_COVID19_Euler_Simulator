import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Fungsi Model dan Euler (Salin dari Jupyter Notebook) ---

# Model: Sistem Persamaan Diferensial Biasa (ODE) SIR
def sir_model(Y, t, N, beta, gamma):
    # ... (Isi dengan kode sir_model dari Cell 2)
    S, I, R = Y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])

def euler_method_sir(func, Y0, t_points, params):
    # ... (Isi dengan kode euler_method_sir dari Cell 3)
    S_points, I_points, R_points = np.zeros(len(t_points)), np.zeros(len(t_points)), np.zeros(len(t_points))
    S_points[0], I_points[0], R_points[0] = Y0
    h = t_points[1] - t_points[0]
    
    for i in range(len(t_points) - 1):
        Y_current = np.array([S_points[i], I_points[i], R_points[i]])
        t_current = t_points[i]
        slopes = func(Y_current, t_current, *params)
        Y_next = Y_current + h * slopes
        S_points[i+1], I_points[i+1], R_points[i+1] = Y_next
        
    return S_points, I_points, R_points

# --- 2. Data Preparation Function ---

@st.cache_data
def load_and_preprocess_data(N):
    # Asumsikan file CSV sudah ada di repositori GitHub
    # Pastikan nama file sama persis
    df = pd.read_csv("time-series-19-covid-combined.csv") 

    df_global = df.groupby('Date')[['Confirmed', 'Recovered', 'Deaths']].sum().reset_index()
    df_global['Active_Infected'] = df_global['Confirmed'] - df_global['Recovered'].fillna(0) - df_global['Deaths']
    df_global['Removed'] = df_global['Recovered'].fillna(0) + df_global['Deaths']
    
    df_sim = df_global.head(300).copy()
    df_sim['Time_Days'] = np.arange(len(df_sim))
    
    I0 = df_sim['Active_Infected'].iloc[0]
    R0 = df_sim['Removed'].iloc[0]
    S0 = N - I0 - R0
    Y0 = np.array([S0, I0, R0])
    
    return df_sim, Y0

# --- 3. Streamlit App Layout ---

st.title("🦠 SIR Model Simulator (Metode Euler)")
st.subheader("Simulasi Penyebaran Penyakit Berdasarkan Data COVID-19")

# --- Sidebar Input Parameter ---
st.sidebar.header("Konfigurasi Parameter Simulasi")

# Total Populasi (N) - Nilai Awal untuk N_pop
N_pop = st.sidebar.number_input("Total Populasi (N)", value=7800000000, step=1000000000, format="%d", help="Asumsi Populasi Global/Area Simulasi")

# Parameter Model (beta dan gamma)
beta = st.sidebar.slider("Laju Penularan (β)", min_value=0.01, max_value=0.5, value=0.25, step=0.01)
gamma = st.sidebar.slider("Laju Pemulihan (γ)", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

# Parameter Numerik (h dan Durasi)
h = st.sidebar.slider("Step Size (h)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Ukuran langkah waktu (Hari). Lebih kecil, lebih akurat.")
durasi = st.sidebar.number_input("Durasi Simulasi (Hari)", value=300, min_value=50, max_value=700, step=50)

# --- Main Logic ---

df_data, Y0 = load_and_preprocess_data(N_pop)
t_points = np.arange(0, durasi + h, h)
params_euler = (N_pop, beta, gamma)

# Jalankan Simulasi Euler
S_euler, I_euler, R_euler = euler_method_sir(sir_model, Y0, t_points, params_euler)

# --- 4. Visualisasi Interaktif ---

st.header("Visualisasi Kurva")
fig, ax = plt.subplots(figsize=(10, 5))

# Plot Simulasi Euler
ax.plot(t_points[:len(I_euler)], I_euler, label='Euler I (Terinfeksi)', color='red')
ax.plot(t_points[:len(S_euler)], S_euler, label='Euler S (Rentan)', color='blue')
ax.plot(t_points[:len(R_euler)], R_euler, label='Euler R (Sembuh/Hilang)', color='green')

# Plot Data Asli (Validasi)
ax.scatter(df_data['Time_Days'], df_data['Active_Infected'], label='Data Asli I (Kasus Aktif)', color='red', marker='o', s=10)

ax.set_title(f'Simulasi SIR dengan Metode Euler (h={h})')
ax.set_xlabel('Waktu (Hari)')
ax.set_ylabel('Jumlah Individu')
ax.legend()
ax.grid(True)
st.pyplot(fig)


# --- 5. Metrik Error ---

st.header("Metrik Error (Validasi)")

# Data yang akan dibandingkan (hanya data dalam rentang simulasi)
max_t = int(min(durasi, len(df_data) - 1))
I_data_val = df_data['Active_Infected'].head(max_t).values

# Ambil hasil simulasi yang sesuai dengan panjang data asli
I_euler_val = I_euler[0:len(I_data_val)]

# Hitung Mean Squared Error (MSE)
if len(I_data_val) == len(I_euler_val):
    mse = np.mean((I_data_val - I_euler_val)**2)
    st.markdown(f"**Mean Squared Error (MSE)** untuk Kasus Aktif selama {max_t} Hari: **{mse:,.0f}**")
    st.markdown("💡 *Geser slider β dan γ untuk meminimalkan nilai MSE (agar kurva simulasi 'fit' dengan data asli).*")
else:
    st.warning("Data simulasi tidak sinkron dengan data asli. Sesuaikan durasi simulasi.")

st.dataframe(df_data[['Time_Days', 'Active_Infected', 'Removed']].head())