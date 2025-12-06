# 🏗️ HVAC Decision Support System (Streamlit App)

**A Building Services Engineering analysis tool compliant with CIBSE Guide A & B, ASHRAE, and Part L Regulations.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📖 Overview
This application serves as a **Decision Support System (DSS)** for HVAC engineers and students. Developed as part of the **TU825 Building Engineering** coursework at TU Dublin, it automates complex psychrometric, ventilation, and duct sizing calculations that are traditionally done via spreadsheets or manual lookups.

The tool bridges the gap between theoretical standards (CIBSE/ASHRAE) and rapid digital prototyping.

## 🚀 Key Features

### 1. 🌬️ Air Change Rates & IAQ
- **Dynamic Sizing:** Calculates fresh air rates based on Occupancy, Floor Area, or Target ACH.
- **CO₂ Simulation:** Solves differential equations to model CO₂ concentration buildup over time.
- **Compliance:** Checks against CIBSE Guide A limits (1000ppm).

### 2. ⚡ Specific Fan Power (SFP)
- **Part L Validation:** Real-time checking against Building Regulations (Limit: 1.5 W/l/s).
- **Lifecycle Costing:** Estimates annual energy cost and carbon footprint based on operating hours.

### 3. 📏 Advanced Duct Sizing
- **Rectangular & Circular:** Supports both geometries using **Huebscher's Equation** for equivalent diameters.
- **Pressure Drop Engine:** Uses the **Swamee-Jain** equation (approximation of Colebrook-White) to calculate precise friction factors and pressure drops (Pa/m).
- **Visualization:** Generates 2D schematic plans and 3D routing sketches of the duct layout.

### 4. 🌡️ Psychrometrics & Thermal Comfort
- **Process Loads:** Calculates Heating/Cooling coil loads (kW) and moisture removal rates.
- **Interactive Chart:** Plots state points and process lines on a simplified psychrometric chart.
- **Thermal Comfort:** Implements **Fanger's PMV/PPD** model (ISO 7730) to assess occupant comfort.

### 5. ⚙️ Fan Laws
- Predicts performance changes (Flow, Pressure, Power) based on RPM adjustments.
- Highlights the cubic relationship between Speed and Power.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/hvac-decision-support.git](https://github.com/yourusername/hvac-decision-support.git)
