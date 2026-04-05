# 📐 Opciones · Superficie de Volatilidad Implícita

> Análisis de la volatilidad implícita con modelos SVI, SABR y señales de reversión a la media AR(1).  
> Datos completamente sintéticos.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Descripción General

Este dashboard implementa un workflow completo de superficie de volatilidad implícita desde cero, sin depender de librerías externas de pricing de opciones. Construido sobre una cadena de opciones, muestra:

| Tab | Qué hace |
|-----|---------|
| **01 · Cadena de Opciones** | Vista completa — IV, griegas, bid/ask spread, volumen por strike |
| **02 · Smile & Modelos de Volatilidad** | Ajustes SVI, SABR (Hagan) y smile cuadrático con análisis de residuos |
| **03 · Reversión a la Media AR(1)** | Detecta opciones baratas/caras estadísticamente vía persistencia de desvíos en IV |

---

## Funcionalidades

### Black-Scholes (`core/bs.py`)
- Pricing completo: `bs_price`, `bs_delta`, `bs_gamma`, `bs_vega`, `bs_theta`, `bs_charm`, `bs_vanna`
- **Volatilidad implícita** (Via Brentq)
- Probabilidad riesgo-neutral de ejercicio: `N(d2)` / `N(-d2)`

### Modelos de Volatilidad (`core/models.py`)

**SVI (Stochastic Volatility Inspired)**
- Parametrización: `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`
- Ajuste con L-BFGS-B sobre varianza total (ponderado por vega)

**SABR (Hagan et al. 2002)**
- Aproximación lognormal (`β = 1.0` por defecto, configurable)
- Parámetros: `α` (nivel de volatilidad), `ρ` (skew), `ν` (vol-of-vol)

**Smile Cuadrático**
- WLS sobre log-moneyness: `iv(k) = a₀ + a₁k + a₂k²`
- Base para extracción de residuos en AR(1)

### Reversión a la Media AR(1) (`Tab 03`)
- Construye panel intradiario de residuos: `ε = IV − IV_quad` por opción
- Ajuste: `εₜ = φ·εₜ₋₁ + ηₜ` vía OLS
- Estima **half-life**: `ln(0.5)/ln(|φ|)`
- Señales por percentiles:
  - `pct < 15%` → LONG  
  - `pct > 85%` → SHORT
- Score de prioridad: `|ε| × vega` (mayor peso a opciones líquidas)

### Generador de Datos Sintéticos (`core/data_gen.py`)
- Cadena tipo GGAL: 29 strikes × 2 vencimientos × C/P
- Smile parametrizado: `iv(k) = ATM + skew·k + smile·k²` + ruido
- 8 días × 60 snapshots intradiarios para estimación AR(1)
- Spread bid/ask creciente con OTM y menor tiempo a vencimiento

---

## Estructura
```
options-iv-surface/
├── app.py # Entrada principal de Streamlit
├── requirements.txt
├── .streamlit/
│ └── config.toml # Configuración tema oscuro
├── core/
│ ├── __init__.py
│ ├── bs.py # Black-Scholes: pricing + griegas + IV
│ ├── models.py # SVI, SABR, Cuadrático, AR(1)
│ └── data_gen.py # Generador de datos sintéticos
└── README.md
```


---

## Inicio 

```bash
git clone https://github.com/yourname/options-iv-surface
cd options-iv-surface

python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

streamlit run app.py
```

Abrir `http://localhost:8501` 


---

## Deploy en Streamlit Cloud
Forkear el repo
Ir a https://share.streamlit.io → New app
Seleccionar app.py como entry point
Click en Deploy