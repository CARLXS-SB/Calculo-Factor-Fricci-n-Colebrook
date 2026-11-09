import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Calculadora Hidráulica - Colebrook-White", layout="wide"
)

st.title("Aplicación de Métodos Numéricos en Ingeniería Civil")
st.subheader("Cálculo del Factor de Fricción ($f$) mediante el Método de Newton-Raphson")

st.markdown("""
Esta aplicación resuelve la ecuación implícita de **Colebrook-White** para flujo turbulentos en tuberías presurizadas:
$$\\frac{1}{\\sqrt{f}} = -2 \\log_{10} \\left( \\frac{\\varepsilon / D}{3.7} + \\frac{2.51}{Re \\sqrt{f}} \\right)$$
Dado que $f$ está en ambos lados, se utiliza el método numérico **Newton-Raphson** para encontrar la raíz de manera iterativa.
""")

st.sidebar.header("Parámetros de Entrada")

# Sliders y entradas en la barra lateral
flow_rate = st.sidebar.slider(
    "Caudal ($Q$ en $m^3/s$)", 0.01, 1.00, 0.15, 0.01
)
diameter = st.sidebar.slider("Diámetro de la tubería ($D$ en m)", 0.05, 1.50, 0.30, 0.05)
roughness = st.sidebar.number_input(
    "Rugosidad absoluta ($\\varepsilon$ en mm)", value=0.045, format="%.4f"
)  # Acero comercial por defecto
viscosity = st.sidebar.number_input(
    r"Viscosidad cinemática ($\nu$ en $m^2/s \times 10^{-6}$)",
    value=1.004,
    format="%.3f",
) * 1e-6

# 1. Cálculos preliminares de mecánica de fluidos
Area = (np.pi * diameter**2) / 4
Velocity = flow_rate / Area
Re = (Velocity * diameter) / viscosity
eps = roughness * 1e-3  # Convertir mm a metros
rel_roughness = eps / diameter


# 2. Función de Newton-Raphson para Colebrook-White
def colebrook_newton(Re, rel_roughness, tol=1e-6, max_iter=100):
    # Estimación inicial usando la ecuación de Swamee-Jain
    f = 0.25 / (
        np.log10((rel_roughness / 3.7) + (5.74 / (Re**0.9)))
    ) ** 2

    history = [f]

    for i in range(max_iter):
        inv_sqrt_f = 1.0 / np.sqrt(f)
        # Ecuación de Colebrook reescrita como F(f) = 0
        term1 = rel_roughness / 3.7
        term2 = 2.51 / (Re * np.sqrt(f))
        F = inv_sqrt_f + 2.0 * np.log10(term1 + term2)

        # Derivada analítica dF/df
        d_term2_df = (
            -2.51 / (Re * 2 * (f**1.5))
        )  # simplificada para la derivada aproximada
        # Derivada usando regla de la cadena para log10(x): 1/(x * ln(10)) * derivada_de_x
        dF_df = -0.5 * (f**-1.5) + 2.0 / (
            (term1 + term2) * np.log(10)
        ) * d_term2_df

        f_new = f - (F / dF_df)

        history.append(f_new)

        if abs(f_new - f) < tol:
            break
        f = f_new

    return f, history


# Ejecutar el método numérico
f_final, iter_history = colebrook_newton(Re, rel_roughness)

# Mostrar resultados principales en columnas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Velocidad del flujo", f"{Velocity:.2f} m/s")
col2.metric("Número de Reynolds ($Re$)", f"{Re:,.0f}")
col3.metric("Tipo de Flujo", "Turbulento" if Re > 4000 else "Laminar/Transición")
col4.metric("Factor de Fricción ($f$)", f"{f_final:.5f}")

st.divider()

# Gráficas de convergencia del método numérico
st.subheader("Convergencia del Método de Newton-Raphson")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    range(len(iter_history)),
    iter_history,
    marker="o",
    color="#1f77b4",
    linewidth=2,
)
ax.set_title("Evolución del factor de fricción por iteración")
ax.set_xlabel("Número de Iteración")
ax.set_ylabel("Valor de $f$")
ax.grid(True, linestyle="--", alpha=0.6)

st.pyplot(fig)
