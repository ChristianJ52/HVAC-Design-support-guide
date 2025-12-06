"""
HVAC Decision Support System
Building Services Engineering Tool
Compliant with CIBSE Guide A & B Standards
Author: TU Dublin Building Engineering Student (TU825)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import pi, log, log10, exp, sqrt

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class EngineeringConstants:
    """Engineering constants and standards"""
    # Air properties at standard conditions
    AIR_DENSITY_STD = 1.2  # kg/m³ at 20°C
    SPECIFIC_HEAT_AIR = 1.006  # kJ/kg·K
    
    # CIBSE Guide A recommendations
    FRESH_AIR_RATE_OFFICE = 10  # l/s/person
    FRESH_AIR_RATE_CLASSROOM = 8  # l/s/person
    FRESH_AIR_RATE_MEETING = 12  # l/s/person
    
    # CO2 concentration levels (ppm)
    CO2_OUTDOOR = 400  # ppm
    CO2_PERSON_GENERATION = 0.0052  # l/s per person
    CO2_ACCEPTABLE_LIMIT = 1000  # ppm (CIBSE Guide A)
    CO2_POOR_LIMIT = 1500  # ppm
    
    # Part L Building Regulations
    SFP_LIMIT_PART_L = 1.5  # W/(l/s) - Maximum for good practice
    SFP_EXCELLENT = 1.0  # W/(l/s) - Excellent efficiency
    SFP_ACCEPTABLE = 1.3  # W/(l/s) - Acceptable
    
    # Standard duct sizes (circular, mm diameter)
    STANDARD_DUCT_SIZES = [
        100, 125, 150, 160, 200, 250, 315, 355, 400, 450, 500, 
        560, 630, 710, 800, 900, 1000, 1120, 1250, 1400, 1600
    ]
    
    # Duct velocity recommendations (m/s)
    DUCT_VELOCITY_MIN = 3.0  # m/s
    DUCT_VELOCITY_MAX_RESIDENTIAL = 6.0  # m/s
    DUCT_VELOCITY_MAX_COMMERCIAL = 8.0  # m/s
    DUCT_VELOCITY_MAX_INDUSTRIAL = 12.0  # m/s

# ============================================================================
# PSYCHROMETRIC CALCULATIONS
# ============================================================================

class PsychrometricCalculator:
    """Psychrometric calculations based on CIBSE/ASHRAE standards"""
    
    @staticmethod
    def saturation_pressure(T):
        """Calculate saturation vapor pressure (Pa) using Magnus formula"""
        # Valid for -20°C to 50°C
        return 610.78 * exp((17.27 * T) / (T + 237.3))
    
    @staticmethod
    def vapor_pressure(T, RH):
        """Calculate vapor pressure from temperature and RH"""
        Psat = PsychrometricCalculator.saturation_pressure(T)
        return (RH / 100.0) * Psat
    
    @staticmethod
    def moisture_content(T, RH, P=101325):
        """Calculate moisture content (g water/kg dry air)"""
        Pv = PsychrometricCalculator.vapor_pressure(T, RH)
        return 0.622 * (Pv / (P - Pv)) * 1000  # Convert to g/kg
    
    @staticmethod
    def dew_point(T, RH):
        """Calculate dew point temperature (°C)"""
        Pv = PsychrometricCalculator.vapor_pressure(T, RH)
        if Pv <= 0:
            return None
        a = 17.27
        b = 237.3
        alpha = log(Pv / 610.78)
        return (b * alpha) / (a - alpha)
    
    @staticmethod
    def enthalpy(T, RH, P=101325):
        """Calculate enthalpy (kJ/kg dry air)"""
        w = PsychrometricCalculator.moisture_content(T, RH, P) / 1000  # Convert to kg/kg
        h = 1.006 * T + w * (2501 + 1.86 * T)  # kJ/kg dry air
        return h
    
    @staticmethod
    def relative_humidity_from_dew_point(T, Tdew):
        """Calculate RH from dry bulb and dew point"""
        Psat_T = PsychrometricCalculator.saturation_pressure(T)
        Psat_Tdew = PsychrometricCalculator.saturation_pressure(Tdew)
        return 100 * (Psat_Tdew / Psat_T)
    
    @staticmethod
    def pmv_ppd(ta, tr, vel, rh, met=1.2, clo=0.5, wme=0):
        """
        Predicted Mean Vote (PMV) and PPD per ISO 7730 (Fanger).
        Returns tuple (pmv, ppd).
        """
        pa = rh * 10 * exp(16.6536 - 4030.183 / (ta + 235))  # water vapor partial pressure Pa
        icl = 0.155 * clo  # clothing insulation m2K/W
        m = met * 58.15  # metabolic rate W/m2
        w = wme * 58.15
        mw = m - w
        if icl <= 0:
            f_cl = 1.0
        else:
            f_cl = 1.0 + 0.31 * icl
        hcf = 12.1 * sqrt(vel)
        taa = ta + 273
        tra = tr + 273
        t_cl = (35.5 - 0.028 * mw) - icl * ((mw / 3.96) - 0.1 - pa)
        for _ in range(150):
            hcn = 2.38 * abs(100.0 * f_cl * (t_cl - ta)) ** 0.25
            hc = max(hcf, hcn)
            t_cl_old = t_cl
            t_cl = ((35.5 - 0.028 * mw) - (icl * (mw - 3.05 * (5.733 - 0.007 * mw - pa) - 0.42 * (mw - 58.15) - 0.0173 * m * (5.867 - pa) - 0.0014 * m * (34 - ta)))) / (1 + 0.155 * icl * f_cl * hc)
            if abs(t_cl - t_cl_old) < 0.001:
                break
        # heat loss terms
        hl1 = 3.05 * max(0, 5733 - 6.99 * mw - pa) / 1000
        hl2 = 0.42 * (mw - 58.15) / 1000
        hl3 = 1.7e-5 * m * (5867 - pa)
        hl4 = 0.0014 * m * (34 - ta)
        hl5 = 3.96e-8 * f_cl * ((t_cl + 273) ** 4 - (tra) ** 4)
        hl6 = f_cl * hc * (t_cl - ta)
        pmv = (0.303 * exp(-0.036 * m) + 0.028) * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100 - 95 * exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
        return pmv, ppd

# ============================================================================
# VENTILATION CALCULATIONS
# ============================================================================

class VentilationCalculator:
    """Ventilation and IAQ calculations"""
    
    @staticmethod
    def fresh_air_requirement(occupancy, rate_per_person, room_volume=None):
        """
        Calculate fresh air requirement
        Returns: flow rate (l/s), ACH if room volume provided
        """
        flow_rate = occupancy * rate_per_person  # l/s
        ach = None
        if room_volume:
            flow_rate_m3s = flow_rate / 1000  # Convert to m³/s
            ach = (flow_rate_m3s * 3600) / room_volume  # Air changes per hour
        return flow_rate, ach
    
    @staticmethod
    def co2_buildup_over_time(room_volume, occupancy, ventilation_rate, 
                              time_hours=8, dt=0.1):
        """
        Model CO2 concentration buildup over time
        Returns: time array (hours), CO2 concentration array (ppm)
        """
        C_out = EngineeringConstants.CO2_OUTDOOR  # ppm
        G = occupancy * EngineeringConstants.CO2_PERSON_GENERATION * 1000  # ml/s
        Q = ventilation_rate / 1000  # m³/s
        V = room_volume  # m³
        
        # Differential equation: dC/dt = (G + Q*C_out - Q*C) / V
        # Analytical solution: C(t) = C_eq + (C0 - C_eq) * exp(-Q*t/V)
        # where C_eq = (G + Q*C_out) / Q
        
        if Q == 0:  # No ventilation
            # Linear buildup: C(t) = C_out + (G/V)*t
            time_array = np.arange(0, time_hours, dt)
            co2_array = C_out + (G / V) * time_array * 3600
        else:
            C_eq = (G + Q * C_out) / Q
            C0 = C_out
            time_array = np.arange(0, time_hours, dt)
            time_seconds = time_array * 3600
            co2_array = C_eq + (C0 - C_eq) * np.exp(-Q * time_seconds / V)
        
        return time_array, co2_array
    
    @staticmethod
    def interpret_ach(ach, space_type="office"):
        """Provide engineering interpretation of ACH value"""
        interpretations = {
            "office": {
                "poor": (0, 2, "⚠️ Severely inadequate - Risk of IAQ issues"),
                "below": (2, 4, "⚡ Below recommended - Consider increasing ventilation"),
                "adequate": (4, 8, "✓ Adequate for general office use"),
                "good": (8, 12, "✓✓ Good ventilation level"),
                "excellent": (12, float('inf'), "✓✓✓ Excellent - May be over-ventilated")
            }
        }
        
        ranges = interpretations.get(space_type, interpretations["office"])
        for key, (low, high, msg) in ranges.items():
            if low <= ach < high:
                return msg
        return "Unknown range"

# ============================================================================
# FAN & DUCT CALCULATIONS
# ============================================================================

class FanCalculator:
    """Fan power and SFP calculations"""
    
    @staticmethod
    def specific_fan_power(power_watts, flow_rate_ls):
        """Calculate Specific Fan Power (W/(l/s))"""
        if flow_rate_ls <= 0:
            return None
        return power_watts / flow_rate_ls
    
    @staticmethod
    def fan_power(flow_rate_m3s, pressure_drop_pa, efficiency=0.7):
        """Calculate theoretical fan power (W)"""
        return (flow_rate_m3s * pressure_drop_pa) / efficiency
    
    @staticmethod
    def interpret_sfp(sfp):
        """Provide compliance interpretation of SFP"""
        if sfp < EngineeringConstants.SFP_EXCELLENT:
            return "🌟 Excellent", "Highly efficient system - exceeds best practice"
        elif sfp < EngineeringConstants.SFP_ACCEPTABLE:
            return "✓ Good", "Good efficiency - acceptable for most applications"
        elif sfp < EngineeringConstants.SFP_LIMIT_PART_L:
            return "⚡ Acceptable", "Meets Part L regulations but could be improved"
        else:
            return "❌ Non-Compliant", "Exceeds Part L limit - system redesign recommended"

class DuctCalculator:
    """Duct sizing calculations"""
    
    @staticmethod
    def calculate_diameter(flow_rate_ls, velocity_ms):
        """
        Calculate required duct diameter (mm)
        flow_rate_ls: flow rate in l/s
        velocity_ms: velocity in m/s
        """
        flow_rate_m3s = flow_rate_ls / 1000
        area_m2 = flow_rate_m3s / velocity_ms
        diameter_m = sqrt((4 * area_m2) / pi)
        diameter_mm = diameter_m * 1000
        return diameter_mm
    
    @staticmethod
    def find_nearest_standard_size(diameter_mm):
        """Find nearest standard duct size"""
        sizes = EngineeringConstants.STANDARD_DUCT_SIZES
        nearest = min(sizes, key=lambda x: abs(x - diameter_mm))
        return nearest
    
    @staticmethod
    def actual_velocity(flow_rate_ls, diameter_mm):
        """Calculate actual velocity for given diameter"""
        diameter_m = diameter_mm / 1000
        area_m2 = pi * (diameter_m ** 2) / 4
        flow_rate_m3s = flow_rate_ls / 1000
        return flow_rate_m3s / area_m2
    
    @staticmethod
    def hydraulic_diameter_rect(width_mm, height_mm):
        """Hydraulic diameter for rectangular ducts (m) per CIBSE/ASHRAE"""
        a = width_mm / 1000
        b = height_mm / 1000
        if (a + b) == 0:
            return None
        return (2 * a * b) / (a + b)
    
    @staticmethod
    def equivalent_circular_diameter(width_mm, height_mm):
        """
        Huebscher equivalent circular diameter (mm) for rectangular ducts.
        Reference: CIBSE/ASHRAE, Huebscher correlation.
        """
        a = width_mm / 1000
        b = height_mm / 1000
        if (a + b) == 0:
            return None
        d_e = (1.30 * (a * b) ** 0.625) / ((a + b) ** 0.25)
        return d_e * 1000
    
    @staticmethod
    def friction_factor_swamee_jain(reynolds, diameter_m, roughness_m=9e-5):
        """
        Swamee-Jain explicit approximation of Colebrook-White.
        Reference: ASHRAE/standard hydraulics. Valid for turbulent Re > ~4000.
        """
        if reynolds <= 0 or diameter_m <= 0:
            return None
        if reynolds < 2300:
            return 64 / reynolds  # laminar
        term = (roughness_m / (3.7 * diameter_m)) + (5.74 / (reynolds ** 0.9))
        try:
            return 0.25 / (log10(term) ** 2)
        except (ValueError, ZeroDivisionError):
            return None
    
    @staticmethod
    def pressure_drop_darcy(flow_rate_ls, hydraulic_diameter_m, area_m2, density=1.2, viscosity=1.81e-5, roughness_m=9e-5):
        """
        Darcy-Weisbach pressure drop (Pa/m) using Swamee-Jain friction factor.
        Returns tuple: (pressure_drop_per_m, friction_factor, reynolds, velocity).
        Reference: Darcy-Weisbach, Swamee-Jain (ASHRAE Fundamentals).
        """
        if hydraulic_diameter_m is None or hydraulic_diameter_m <= 0 or area_m2 <= 0:
            return None, None, None, None
        flow_m3s = flow_rate_ls / 1000
        velocity = flow_m3s / area_m2
        if velocity <= 0 or viscosity <= 0:
            return None, None, None, None
        reynolds = (density * velocity * hydraulic_diameter_m) / viscosity
        f = DuctCalculator.friction_factor_swamee_jain(reynolds, hydraulic_diameter_m, roughness_m)
        if f is None:
            return None, None, reynolds, velocity
        dp_per_m = f * (density * velocity ** 2) / (2 * hydraulic_diameter_m)
        return dp_per_m, f, reynolds, velocity
    
    @staticmethod
    def interpret_velocity(velocity, application="commercial"):
        """Interpret if velocity is appropriate"""
        limits = {
            "residential": EngineeringConstants.DUCT_VELOCITY_MAX_RESIDENTIAL,
            "commercial": EngineeringConstants.DUCT_VELOCITY_MAX_COMMERCIAL,
            "industrial": EngineeringConstants.DUCT_VELOCITY_MAX_INDUSTRIAL
        }
        max_vel = limits.get(application, 8.0)
        
        if velocity < EngineeringConstants.DUCT_VELOCITY_MIN:
            return "⚠️ Too Low", "Risk of dust settlement and poor air distribution"
        elif velocity > max_vel:
            return "❌ Too High", "Excessive noise and pressure drop - increase duct size"
        elif velocity > max_vel * 0.8:
            return "⚡ High", "Near maximum - consider larger size for quieter operation"
        else:
            return "✓ Acceptable", "Velocity within recommended range"

# ============================================================================
# REPORT BUILDER
# ============================================================================

def generate_report(data):
    """
    Build a plain-text report of key HVAC calculations (CIBSE/ASHRAE/ISO refs).
    """
    lines = []
    lines.append("HVAC Decision Support System Report")
    lines.append("-----------------------------------")
    lines.append("Ventilation & IAQ")
    lines.append(f"  Method: {data.get('vent_method', 'N/A')}")
    lines.append(f"  Flow: {data.get('vent_flow_ls', 'N/A')} l/s | ACH: {data.get('vent_ach', 'N/A')} h-1")
    lines.append(f"  CO2 Target: {data.get('vent_co2_target', 'N/A')} ppm")
    if data.get("vent_insights"):
        for ins in data["vent_insights"]:
            lines.append(f"    - {ins}")
    lines.append("")
    lines.append("Fan Power & SFP")
    lines.append(f"  Flow: {data.get('fan_flow_ls', 'N/A')} l/s | SFP: {data.get('sfp', 'N/A')} W/(l/s)")
    lines.append(f"  Fan Power: {data.get('fan_power_w', 'N/A')} W")
    lines.append("")
    lines.append("Duct Sizing")
    lines.append(f"  Shape: {data.get('duct_shape', 'N/A')} | Main: {data.get('duct_main_size_mm', 'N/A')} mm | Branch: {data.get('duct_branch_size_mm', 'N/A')} mm")
    lines.append(f"  Velocities: main {data.get('duct_velocity_main', 'N/A')} m/s, branch {data.get('duct_velocity_branch', 'N/A')} m/s")
    lines.append(f"  Pressure Drop: {data.get('duct_pressure_drop_pa_per_m', 'N/A')} Pa/m | f={data.get('duct_friction_factor', 'N/A')} | Re={data.get('duct_reynolds', 'N/A')}")
    lines.append(f"  Diffusers: {data.get('duct_diffuser_count', 'N/A')}")
    lines.append("")
    lines.append("Psychrometrics")
    lines.append(f"  Dry Bulb: {data.get('psychro_dry_bulb', 'N/A')} °C | RH: {data.get('psychro_rh', 'N/A')} %")
    lines.append(f"  Enthalpy: {data.get('psychro_enthalpy', 'N/A')} kJ/kg | Moisture: {data.get('psychro_moisture', 'N/A')} g/kg | Dew Point: {data.get('psychro_dew_point', 'N/A')} °C")
    lines.append(f"  PMV: {data.get('pmv', 'N/A')} | PPD: {data.get('ppd', 'N/A')} %")
    lines.append(f"  Heating Load: {data.get('heating_power_kw', 'N/A')} kW | Cooling Load: {data.get('cooling_power_kw', 'N/A')} kW")
    lines.append("")
    lines.append("Fan Laws Projection")
    lines.append(f"  New RPM: {data.get('fan_law_rpm_target', 'N/A')} | Flow: {data.get('fan_law_flow', 'N/A')} l/s | Pressure: {data.get('fan_law_pressure', 'N/A')} Pa | Power: {data.get('fan_law_power', 'N/A')} W")
    lines.append("")
    lines.append("Notes: Values are indicative. Verify against project-specific standards (CIBSE Guide A/B, ASHRAE, ISO 7730).")
    return "\n".join(str(x) for x in lines)

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    # Initialize shared report data
    if "report_data" not in st.session_state:
        st.session_state["report_data"] = {}
    report_data = st.session_state["report_data"]
    report_data.clear()
    # Page configuration
    st.set_page_config(
        page_title="HVAC Decision Support System",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .insight-box {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2ecc71;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #dc3545;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🏗️ HVAC Decision Support System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Building Services Engineering Analysis Tool | CIBSE Compliant</p>', unsafe_allow_html=True)
    
    # Sidebar - Global Settings
    with st.sidebar:
        st.header("⚙️ Global Settings")
        
        st.subheader("Environmental Conditions")
        altitude = st.slider("Altitude (m)", 0, 2000, 0, 50, 
                            help="Affects air density and psychrometric calculations")
        
        # Calculate atmospheric pressure based on altitude
        P_atm = 101325 * (1 - 2.25577e-5 * altitude) ** 5.25588
        
        ambient_temp = st.number_input("Ambient Temperature (°C)", 
                                       value=20.0, min_value=-20.0, max_value=50.0)
        
        # Air density adjustment
        air_density = EngineeringConstants.AIR_DENSITY_STD * (P_atm / 101325) * (293.15 / (ambient_temp + 273.15))
        
        st.metric("Air Density", f"{air_density:.3f} kg/m³")
        st.metric("Atmospheric Pressure", f"{P_atm/1000:.2f} kPa")
        
        st.markdown("---")
        st.subheader("About")
        st.info("""
        **TU Dublin - TU825**  
        Building Engineering Student Project
        
        Compliant with:
        - CIBSE Guide A (Environmental Design)
        - CIBSE Guide B (Heating Systems)
        - Part L Building Regulations
        """)
        
        st.markdown("---")
        generate_now = st.button("📄 Generate Report")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌬️ Air Change Rates & IAQ",
        "⚡ SFP & Fan Power",
        "📏 Duct Sizing",
        "🌡️ Psychrometrics",
        "⚙️ Fan Laws"
    ])
    
    # ========================================================================
    # TAB 1: AIR CHANGE RATES & IAQ
    # ========================================================================
    with tab1:
        st.header("Air Change Rates & Indoor Air Quality")
        st.markdown("*Calculate fresh air requirements and model CO₂ buildup per CIBSE Guide A*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Room Parameters")
            
            room_length = st.number_input("Room Length (m)", value=10.0, min_value=1.0, step=0.5)
            room_width = st.number_input("Room Width (m)", value=8.0, min_value=1.0, step=0.5)
            room_height = st.number_input("Room Height (m)", value=3.0, min_value=2.0, step=0.1)
            room_volume = room_length * room_width * room_height
            
            st.metric("Room Volume", f"{room_volume:.1f} m³")
            
            occupancy = st.number_input("Number of Occupants", value=20, min_value=1, step=1)
            
            space_type = st.selectbox(
                "Space Type",
                ["Office", "Classroom", "Meeting Room", "Custom"],
                help="Pre-defined fresh air rates per CIBSE Guide A"
            )
            
            if space_type == "Office":
                default_rate = EngineeringConstants.FRESH_AIR_RATE_OFFICE
            elif space_type == "Classroom":
                default_rate = EngineeringConstants.FRESH_AIR_RATE_CLASSROOM
            elif space_type == "Meeting Room":
                default_rate = EngineeringConstants.FRESH_AIR_RATE_MEETING
            else:
                default_rate = 10.0
            
            fresh_air_rate = st.number_input(
                "Fresh Air Rate (l/s/person)",
                value=float(default_rate),
                min_value=0.0,
                step=0.5,
                help="CIBSE Guide A recommends 8-12 l/s/person for offices"
            )
            
            area_fresh_air_rate = st.number_input(
                "Area-Based Fresh Air (l/s/m²)",
                value=1.0,
                min_value=0.0,
                step=0.1,
                help="Use for standards that set outdoor air per floor area"
            )
            
            target_ach_input = st.number_input(
                "Target ACH (h⁻¹)",
                value=6.0,
                min_value=0.0,
                step=0.5,
                help="Alternative sizing basis using target air changes per hour"
            )
            
            target_co2 = st.number_input(
                "Target Indoor CO₂ (ppm)",
                value=900,
                min_value=EngineeringConstants.CO2_OUTDOOR + 50,
                step=25,
                help="Demand-control option: solves required outdoor air to hit this steady-state setpoint"
            )
        
        with col2:
            st.subheader("Ventilation Analysis")
            
            # Calculate multiple sizing methods
            flow_occ = occupancy * fresh_air_rate  # l/s
            flow_area = (room_length * room_width) * area_fresh_air_rate  # l/s
            flow_ach = (target_ach_input * room_volume) / 3.6  # l/s
            flow_max_occ_area = max(flow_occ, flow_area)
            
            co2_flow = None
            if target_co2 > EngineeringConstants.CO2_OUTDOOR:
                generation_ls = occupancy * EngineeringConstants.CO2_PERSON_GENERATION  # l/s of CO2
                delta_ppm = max(target_co2 - EngineeringConstants.CO2_OUTDOOR, 1e-6)
                co2_flow = (generation_ls * 1e6) / delta_ppm  # l/s required to hold target
            
            sizing_options = {
                "Occupant Basis": flow_occ,
                "Area Basis": flow_area,
                "ACH Basis": flow_ach,
                "Max of Occupant & Area": flow_max_occ_area
            }
            if co2_flow is not None:
                sizing_options["CO₂ Setpoint"] = co2_flow
            
            sizing_method = st.selectbox(
                "Design Method",
                list(sizing_options.keys()),
                help="Choose which sizing basis to use for downstream calculations"
            )
            
            total_flow_rate = sizing_options[sizing_method]
            ach = (total_flow_rate / 1000) * 3600 / room_volume if room_volume else 0
            
            # Display results
            st.metric("Total Fresh Air Required", f"{total_flow_rate:.1f} l/s")
            st.metric("Air Changes per Hour (ACH)", f"{ach:.2f} h⁻¹")
            report_data.update({
                "vent_flow_ls": total_flow_rate,
                "vent_ach": ach,
                "vent_method": sizing_method,
                "vent_space_type": space_type,
                "vent_co2_target": target_co2,
                "vent_flow_occ": flow_occ,
                "vent_flow_area": flow_area,
                "vent_flow_ach": flow_ach,
            })
            
            # Comparison table of methods
            comparison_rows = []
            for name, flow in sizing_options.items():
                ach_method = (flow / 1000) * 3600 / room_volume if room_volume else 0
                comparison_rows.append({
                    "Method": name,
                    "Flow (l/s)": f"{flow:.1f}",
                    "ACH": f"{ach_method:.2f}"
                })
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, height=170)
            
            # Interpretation
            interpretation = VentilationCalculator.interpret_ach(ach, space_type.lower())
            
            if "⚠️" in interpretation or "⚡" in interpretation:
                st.markdown(f'<div class="warning-box">{interpretation}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box">{interpretation}</div>', unsafe_allow_html=True)
            
            # Additional metrics
            flow_per_area = total_flow_rate / (room_length * room_width)
            occupancy_density = occupancy / (room_length * room_width)
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Flow per Floor Area", f"{flow_per_area:.2f} l/s/m²")
            with col2b:
                st.metric("Occupancy Density", f"{occupancy_density:.2f} people/m²")
        
        st.markdown("---")
        
        # CO2 Buildup Simulation
        st.subheader("CO₂ Concentration Analysis")
        st.markdown("*Model CO₂ buildup over time with different ventilation scenarios*")
        
        col3, col4 = st.columns([1, 2])
        
        with col3:
            simulation_hours = st.slider("Simulation Duration (hours)", 1, 12, 8)
            
            st.markdown("**Ventilation Scenarios:**")
            scenario_design = st.checkbox("Design Ventilation Rate", value=True)
            scenario_reduced = st.checkbox("50% Reduced Ventilation", value=True)
            scenario_none = st.checkbox("No Ventilation (Failure)", value=True)
        
        with col4:
            # Run CO2 simulations
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scenarios = []
            if scenario_design:
                scenarios.append(("Design Rate", total_flow_rate, "green", "-"))
            if scenario_reduced:
                scenarios.append(("50% Reduced", total_flow_rate * 0.5, "orange", "--"))
            if scenario_none:
                scenarios.append(("No Ventilation", 0.0, "red", "-."))
            
            for label, vent_rate, color, linestyle in scenarios:
                time_arr, co2_arr = VentilationCalculator.co2_buildup_over_time(
                    room_volume, occupancy, vent_rate, simulation_hours
                )
                ax.plot(time_arr, co2_arr, label=label, color=color, 
                       linestyle=linestyle, linewidth=2)
            
            # Reference lines
            ax.axhline(y=EngineeringConstants.CO2_ACCEPTABLE_LIMIT, 
                      color='blue', linestyle=':', linewidth=1.5, 
                      label='CIBSE Limit (1000 ppm)', alpha=0.7)
            ax.axhline(y=EngineeringConstants.CO2_POOR_LIMIT, 
                      color='darkred', linestyle=':', linewidth=1.5,
                      label='Poor IAQ (1500 ppm)', alpha=0.7)
            
            ax.set_xlabel("Time (hours)", fontsize=12, fontweight='bold')
            ax.set_ylabel("CO₂ Concentration (ppm)", fontsize=12, fontweight='bold')
            ax.set_title("Indoor CO₂ Concentration Over Time", fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([300, min(3000, max(co2_arr) * 1.2)])
            
            st.pyplot(fig)
        
        # Engineering Insights
        st.markdown("---")
        st.subheader("💡 Engineering Insights & Recommendations")
        
        insights = []
        
        if ach < 4:
            insights.append("❌ **Critical:** ACH is below minimum recommendations. Increase ventilation rate or reduce occupancy.")
        elif ach < 6:
            insights.append("⚡ **Action Required:** Consider increasing fresh air rate to improve IAQ and occupant comfort.")
        
        if total_flow_rate > 500:
            insights.append("ℹ️ **Large System:** This flow rate may require multiple AHUs or a centralized system with VAV control.")
        
        if occupancy_density > 0.15:
            insights.append("👥 **High Density:** Space has high occupancy density. Ensure adequate fresh air supply.")
        
        # Energy consideration
        heating_load_estimate = total_flow_rate / 1000 * air_density * EngineeringConstants.SPECIFIC_HEAT_AIR * (20 - ambient_temp)
        if heating_load_estimate > 5:
            insights.append(f"💰 **Energy Note:** Estimated heating load ~{heating_load_estimate:.1f} kW. Consider heat recovery ventilation (HRV) to reduce energy costs.")
        
        if not insights:
            insights.append("✅ **Compliant:** Ventilation design meets CIBSE standards and provides adequate IAQ.")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Store for report
        report_data.update({
            "vent_insights": insights
        })
    
    # ========================================================================
    # TAB 2: SFP & FAN POWER
    # ========================================================================
    with tab2:
        st.header("Specific Fan Power & Compliance")
        st.markdown("*Calculate SFP and verify compliance with Part L Building Regulations*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("System Parameters")
            
            calculation_method = st.radio(
                "Calculation Method",
                ["Direct Entry (Known Power)", "From Pressure Drop"],
                help="Choose based on available information"
            )
            
            flow_rate_fan = st.number_input(
                "Air Flow Rate (l/s)",
                value=500.0,
                min_value=10.0,
                step=10.0
            )
            
            if calculation_method == "Direct Entry (Known Power)":
                fan_power_input = st.number_input(
                    "Fan Power (W)",
                    value=600.0,
                    min_value=0.0,
                    step=10.0
                )
                sfp = FanCalculator.specific_fan_power(fan_power_input, flow_rate_fan)
            else:
                pressure_drop = st.number_input(
                    "Total System Pressure Drop (Pa)",
                    value=500.0,
                    min_value=0.0,
                    step=50.0,
                    help="Sum of all pressure losses in the system"
                )
                
                fan_efficiency = st.slider(
                    "Fan Total Efficiency (%)",
                    min_value=50,
                    max_value=90,
                    value=70,
                    help="Typical range: 60-80% for commercial fans"
                ) / 100.0
                
                fan_power_calculated = FanCalculator.fan_power(
                    flow_rate_fan / 1000, pressure_drop, fan_efficiency
                )
                fan_power_input = fan_power_calculated
                sfp = FanCalculator.specific_fan_power(fan_power_calculated, flow_rate_fan)
                
                st.metric("Calculated Fan Power", f"{fan_power_calculated:.1f} W")
        
        with col2:
            st.subheader("Results & Compliance")
            
            # Display SFP
            st.metric("Specific Fan Power (SFP)", f"{sfp:.3f} W/(l/s)")
            report_data.update({
                "sfp": sfp,
                "fan_power_w": fan_power_input,
                "fan_flow_ls": flow_rate_fan
            })
            
            # Interpretation
            status, message = FanCalculator.interpret_sfp(sfp)
            
            # Compliance visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            
            categories = ['Excellent\n(<1.0)', 'Good\n(1.0-1.3)', 
                         'Acceptable\n(1.3-1.5)', 'Non-Compliant\n(>1.5)']
            values = [1.0, 1.3, 1.5, 2.0]
            colors = ['green', 'lightgreen', 'orange', 'red']
            
            bars = ax.barh(categories, values, color=colors, alpha=0.6)
            
            # Plot actual SFP
            ax.axvline(x=sfp, color='darkblue', linewidth=3, label=f'Your SFP: {sfp:.3f}')
            ax.axvline(x=EngineeringConstants.SFP_LIMIT_PART_L, 
                      color='black', linewidth=2, linestyle='--', 
                      label='Part L Limit: 1.5', alpha=0.7)
            
            ax.set_xlabel('SFP [W/(l/s)]', fontsize=12, fontweight='bold')
            ax.set_title('SFP Performance vs. Part L Regulations', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_xlim([0, min(2.5, sfp * 1.3)])
            
            st.pyplot(fig)
            
            # Status box
            if "❌" in status:
                st.markdown(f'<div class="error-box"><strong>{status}:</strong> {message}</div>', 
                           unsafe_allow_html=True)
            elif "⚡" in status:
                st.markdown(f'<div class="warning-box"><strong>{status}:</strong> {message}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box"><strong>{status}:</strong> {message}</div>', 
                           unsafe_allow_html=True)
        
        # Additional Analysis
        st.markdown("---")
        st.subheader("System Analysis")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            annual_hours = st.number_input("Annual Operating Hours", value=2500, min_value=0, step=100)
        
        with col4:
            electricity_cost = st.number_input("Electricity Cost (€/kWh)", value=0.15, min_value=0.0, step=0.01)
        
        with col5:
            # Calculate annual energy consumption
            annual_energy_kwh = (fan_power_input * annual_hours) / 1000
            annual_cost = annual_energy_kwh * electricity_cost
            
            st.metric("Annual Energy", f"{annual_energy_kwh:.0f} kWh")
            st.metric("Annual Cost", f"€{annual_cost:.2f}")
        
        # Engineering Insights
        st.markdown("---")
        st.subheader("💡 Engineering Insights & Recommendations")
        
        insights_sfp = []
        
        if sfp > EngineeringConstants.SFP_LIMIT_PART_L:
            improvement_potential = ((sfp - 1.2) / sfp) * annual_cost
            insights_sfp.append(f"❌ **Non-Compliance:** System exceeds Part L regulations. Reducing SFP to 1.2 W/(l/s) could save ~€{improvement_potential:.2f}/year.")
            insights_sfp.append("🔧 **Recommended Actions:** Increase duct sizes, optimize duct layout, use low-pressure drop components, or upgrade to higher efficiency fan.")
        
        if sfp < 1.0:
            insights_sfp.append("🌟 **Excellent Design:** SFP indicates a highly efficient system with low operating costs.")
        
        if calculation_method == "From Pressure Drop":
            if pressure_drop > 600:
                insights_sfp.append("⚠️ **High Pressure Drop:** Consider reviewing duct design. High pressure drops increase fan power and reduce system efficiency.")
            if fan_efficiency < 0.65:
                efficiency_gain = (0.75 - fan_efficiency) / 0.75
                potential_saving = efficiency_gain * annual_cost
                insights_sfp.append(f"⚡ **Fan Efficiency:** Upgrading to a fan with 75% efficiency could save ~€{potential_saving:.2f}/year.")
        
        # Carbon footprint
        carbon_factor = 0.233  # kg CO2/kWh (Ireland grid average 2024)
        annual_co2 = annual_energy_kwh * carbon_factor
        insights_sfp.append(f"🌍 **Carbon Footprint:** Annual emissions ~{annual_co2:.0f} kg CO₂. Consider energy efficiency improvements to reduce environmental impact.")
        
        for insight in insights_sfp:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: DUCT SIZING
    # ========================================================================
    with tab3:
        st.header("Duct Sizing Calculator")
        st.markdown("*Size circular ductwork based on flow rate and velocity constraints*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Parameters")
            
            duct_shape = st.radio(
                "Duct Shape",
                ["Circular", "Rectangular"],
                horizontal=True
            )
            
            duct_flow_rate = st.number_input(
                "Air Flow Rate (l/s)",
                value=250.0,
                min_value=10.0,
                step=10.0
            )
            
            application_type = st.selectbox(
                "Application Type",
                ["Commercial", "Residential", "Industrial"],
                help="Determines maximum allowable velocity"
            )
            
            max_velocity_recommended = {
                "Commercial": EngineeringConstants.DUCT_VELOCITY_MAX_COMMERCIAL,
                "Residential": EngineeringConstants.DUCT_VELOCITY_MAX_RESIDENTIAL,
                "Industrial": EngineeringConstants.DUCT_VELOCITY_MAX_INDUSTRIAL
            }[application_type]
            
            target_velocity = st.slider(
                "Target Velocity (m/s)",
                min_value=3.0,
                max_value=12.0,
                value=min(6.0, max_velocity_recommended),
                step=0.5,
                help=f"Recommended maximum for {application_type.lower()}: {max_velocity_recommended} m/s"
            )
            
            if duct_shape == "Circular":
                # Calculate required diameter
                required_diameter = DuctCalculator.calculate_diameter(duct_flow_rate, target_velocity)
                nearest_standard = DuctCalculator.find_nearest_standard_size(required_diameter)
                actual_velocity = DuctCalculator.actual_velocity(duct_flow_rate, nearest_standard)
                area_m2 = pi * (nearest_standard / 2000) ** 2
                hydraulic_diameter_m = nearest_standard / 1000
                eq_circ = nearest_standard
                width_mm = height_mm = None
            else:
                width_mm = st.number_input("Rectangular Width (mm)", value=500.0, min_value=100.0, step=25.0)
                height_mm = st.number_input("Rectangular Height (mm)", value=300.0, min_value=100.0, step=25.0)
                area_m2 = (width_mm / 1000) * (height_mm / 1000)
                hydraulic_diameter_m = DuctCalculator.hydraulic_diameter_rect(width_mm, height_mm)
                eq_circ = DuctCalculator.equivalent_circular_diameter(width_mm, height_mm)
                actual_velocity = (duct_flow_rate / 1000) / area_m2 if area_m2 else 0.0
                required_diameter = eq_circ
                nearest_standard = DuctCalculator.find_nearest_standard_size(eq_circ) if eq_circ else None
            
            st.markdown("---")
            st.subheader("Results")
            
            col1a, col1b = st.columns(2)
            with col1a:
                if duct_shape == "Circular":
                    st.metric("Required Diameter", f"{required_diameter:.1f} mm")
                else:
                    st.metric("Equivalent Circular", f"{eq_circ:.1f} mm")
            with col1b:
                if nearest_standard:
                    st.metric("Nearest Circular Size", f"{nearest_standard} mm")
                else:
                    st.metric("Nearest Circular Size", "N/A")
            
            col1c, col1d = st.columns(2)
            with col1c:
                st.metric("Actual Velocity", f"{actual_velocity:.2f} m/s")
            with col1d:
                if duct_shape == "Circular" and nearest_standard:
                    size_difference = ((nearest_standard - required_diameter) / required_diameter) * 100
                    st.metric("Size Adjustment", f"{size_difference:+.1f}%")
                else:
                    st.metric("Size Adjustment", "—")
            
            # Velocity interpretation
            vel_status, vel_message = DuctCalculator.interpret_velocity(actual_velocity, application_type.lower())
            
            if "❌" in vel_status or "⚠️" in vel_status:
                st.markdown(f'<div class="warning-box"><strong>{vel_status}:</strong> {vel_message}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box"><strong>{vel_status}:</strong> {vel_message}</div>', 
                           unsafe_allow_html=True)
            
            # Pressure drop using Darcy-Weisbach + Swamee-Jain
            dp_per_m, f_factor, reynolds, vel_used = DuctCalculator.pressure_drop_darcy(
                duct_flow_rate,
                hydraulic_diameter_m,
                area_m2,
                density=air_density
            )
            
            col_pd1, col_pd2 = st.columns(2)
            with col_pd1:
                st.metric("Pressure Drop", f"{dp_per_m:.2f} Pa/m" if dp_per_m is not None else "N/A")
            with col_pd2:
                st.metric("Friction Factor", f"{f_factor:.4f}" if f_factor is not None else "N/A")
            st.caption("Swamee-Jain friction factor with Darcy-Weisbach pressure drop.")
        
        with col2:
            st.subheader("Standard Duct Sizes Reference")
            
            if duct_shape == "Circular":
                # Create comparison table
                comparison_data = []
                for size in EngineeringConstants.STANDARD_DUCT_SIZES:
                    vel = DuctCalculator.actual_velocity(duct_flow_rate, size)
                    area = pi * (size / 2000) ** 2
                    comparison_data.append({
                        "Diameter (mm)": size,
                        "Velocity (m/s)": f"{vel:.2f}",
                        "Area (m²)": f"{area:.4f}"
                    })
                
                df = pd.DataFrame(comparison_data)
                
                # Highlight the selected size
                def highlight_row(row):
                    if row["Diameter (mm)"] == nearest_standard:
                        return ['background-color: lightgreen'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    df.style.apply(highlight_row, axis=1),
                    height=400,
                    use_container_width=True
                )
                
                st.caption("✅ Highlighted row shows the recommended standard size for your application")
            else:
                st.info("Standard circular size table is hidden for rectangular selection.")
        
        # Velocity Chart
        st.markdown("---")
        st.subheader("Velocity Analysis Across Standard Sizes")
        
        if duct_shape == "Circular":
            fig, ax = plt.subplots(figsize=(12, 5))
            
            sizes_array = np.array(EngineeringConstants.STANDARD_DUCT_SIZES)
            velocities = [DuctCalculator.actual_velocity(duct_flow_rate, s) for s in sizes_array]
            
            # Plot bars
            colors = ['red' if v > max_velocity_recommended or v < EngineeringConstants.DUCT_VELOCITY_MIN 
                     else 'orange' if v > max_velocity_recommended * 0.8 
                     else 'green' for v in velocities]
            
            bars = ax.bar(range(len(sizes_array)), velocities, color=colors, alpha=0.7, edgecolor='black')
            
            # Highlight selected size
            selected_idx = list(sizes_array).index(nearest_standard)
            bars[selected_idx].set_edgecolor('darkblue')
            bars[selected_idx].set_linewidth(3)
            
            # Reference lines
            ax.axhline(y=max_velocity_recommended, color='red', linestyle='--', 
                      linewidth=2, label=f'Max Velocity ({max_velocity_recommended} m/s)', alpha=0.7)
            ax.axhline(y=EngineeringConstants.DUCT_VELOCITY_MIN, color='blue', 
                      linestyle='--', linewidth=2, label=f'Min Velocity ({EngineeringConstants.DUCT_VELOCITY_MIN} m/s)', alpha=0.7)
            
            ax.set_xlabel('Duct Size (mm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
            ax.set_title('Air Velocity for Different Standard Duct Sizes', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(sizes_array)))
            ax.set_xticklabels(sizes_array, rotation=45, ha='right')
            ax.legend(loc='best')
            ax.grid(True, axis='y', alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.info("Velocity bar chart is available for circular duct selections.")
        
        # Engineering Insights
        st.markdown("---")
        st.subheader("💡 Engineering Insights & Recommendations")
        
        insights_duct = []
        
        if actual_velocity > max_velocity_recommended:
            if duct_shape == "Circular" and nearest_standard and nearest_standard != EngineeringConstants.STANDARD_DUCT_SIZES[-1]:
                next_size_up = EngineeringConstants.STANDARD_DUCT_SIZES[
                    EngineeringConstants.STANDARD_DUCT_SIZES.index(nearest_standard) + 1
                ]
                vel_next_size = DuctCalculator.actual_velocity(duct_flow_rate, next_size_up)
                insights_duct.append(f"❌ **Velocity Too High:** Current velocity {actual_velocity:.2f} m/s exceeds {max_velocity_recommended} m/s limit. Use {next_size_up} mm duct (velocity: {vel_next_size:.2f} m/s) for quieter operation.")
            else:
                insights_duct.append(f"❌ **Velocity Too High:** Current velocity {actual_velocity:.2f} m/s exceeds {max_velocity_recommended} m/s limit. Increase duct size or reduce flow per branch.")
        
        if actual_velocity < EngineeringConstants.DUCT_VELOCITY_MIN:
            insights_duct.append(f"⚠️ **Velocity Too Low:** Risk of dust settlement and poor mixing. Consider reducing duct size or using rectangular ducts with better aspect ratios.")
        
        # Noise estimation (rough)
        if actual_velocity > 8:
            insights_duct.append("🔊 **Noise Warning:** High velocity may cause excessive noise. Consider acoustic lining or silencers.")
        
        # Pressure drop estimate (Darcy-Weisbach)
        if dp_per_m is not None:
            insights_duct.append(f"📊 **Pressure Drop:** ~{dp_per_m:.2f} Pa/m (Re={reynolds:.0f}). For 50 m run: ≈ {dp_per_m * 50:.0f} Pa.")
        
        # Area utilization
        if duct_shape == "Circular" and nearest_standard and required_diameter:
            area_utilization = (required_diameter / nearest_standard) ** 2
            if area_utilization < 0.7:
                insights_duct.append(f"⚡ **Over-sized:** Selected duct uses only {area_utilization*100:.0f}% of available area. This is acceptable for low noise but may increase costs.")
        
        if not insights_duct:
            insights_duct.append("✅ **Optimal Design:** Duct size provides appropriate velocity and meets all design constraints.")
        
        for insight in insights_duct:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Duct layout visualization
        st.markdown("---")
        st.subheader("Duct Layout Preview (2D & 3D)")
        
        room_len = room_length if 'room_length' in locals() else 10.0
        room_wid = room_width if 'room_width' in locals() else 8.0
        ceiling_height = room_height if 'room_height' in locals() else 3.0
        
        diffusers_x = st.slider("Diffusers Along Room Length", min_value=1, max_value=6, value=2)
        diffusers_y = st.slider("Diffusers Along Room Width", min_value=1, max_value=6, value=2)
        supply_height = st.number_input(
            "Supply Height (m)",
            value=min(ceiling_height, 3.0),
            min_value=2.0,
            max_value=max(ceiling_height, 2.0),
            step=0.1,
            help="Ceiling or high-level wall discharge height"
        )
        
        diffuser_count = diffusers_x * diffusers_y
        branch_flow = duct_flow_rate / diffuser_count if diffuser_count else 0.0
        
        main_diameter = DuctCalculator.find_nearest_standard_size(
            DuctCalculator.calculate_diameter(duct_flow_rate, target_velocity)
        )
        branch_diameter = DuctCalculator.find_nearest_standard_size(
            DuctCalculator.calculate_diameter(branch_flow, target_velocity)
        )
        
        main_velocity = DuctCalculator.actual_velocity(duct_flow_rate, main_diameter)
        branch_velocity = DuctCalculator.actual_velocity(branch_flow, branch_diameter)
        
        # Position diffusers on a grid
        x_positions = np.linspace(room_len / (diffusers_x + 1), room_len - room_len / (diffusers_x + 1), diffusers_x)
        y_positions = np.linspace(room_wid / (diffusers_y + 1), room_wid - room_wid / (diffusers_y + 1), diffusers_y)
        diffuser_points = [(x, y) for x in x_positions for y in y_positions]
        
        # 2D plan view
        fig_plan, ax_plan = plt.subplots(figsize=(8, 5))
        ax_plan.add_patch(Rectangle((0, 0), room_len, room_wid, fill=False, edgecolor='black', linewidth=1.5))
        ax_plan.plot([0, room_len], [room_wid / 2, room_wid / 2], color='navy', linewidth=3, label=f"Main {main_diameter} mm")
        
        for (x, y) in diffuser_points:
            ax_plan.plot([x, x], [room_wid / 2, y], color='seagreen', linewidth=2, alpha=0.8)
            ax_plan.scatter(x, y, color='orange', s=80, edgecolors='k', zorder=5)
            ax_plan.text(x, y + 0.2, f"{branch_diameter} mm", ha='center', va='bottom', fontsize=8, color='dimgray')
        
        ax_plan.text(room_len * 0.5, room_wid / 2 + 0.2, f"Main {main_diameter} mm | {main_velocity:.1f} m/s", 
                     ha='center', va='bottom', fontsize=9, color='navy')
        ax_plan.set_xlim(0, room_len)
        ax_plan.set_ylim(0, room_wid)
        ax_plan.set_aspect('equal', adjustable='box')
        ax_plan.set_xlabel("Length (m)")
        ax_plan.set_ylabel("Width (m)")
        ax_plan.set_title("Schematic Plan View")
        ax_plan.grid(True, alpha=0.2)
        st.pyplot(fig_plan)
        
        # 3D sketch
        fig_3d = plt.figure(figsize=(8, 5))
        ax3d = fig_3d.add_subplot(111, projection='3d')
        ax3d.plot([0, room_len], [room_wid / 2, room_wid / 2], [supply_height, supply_height], color='navy', linewidth=3)
        for (x, y) in diffuser_points:
            ax3d.plot([x, x], [room_wid / 2, y], [supply_height, supply_height], color='seagreen', linewidth=2, alpha=0.8)
            ax3d.scatter(x, y, supply_height, color='orange', s=40, edgecolors='k', depthshade=True)
        ax3d.set_xlabel("Length (m)")
        ax3d.set_ylabel("Width (m)")
        ax3d.set_zlabel("Height (m)")
        ax3d.set_xlim(0, room_len)
        ax3d.set_ylim(0, room_wid)
        ax3d.set_zlim(0, max(ceiling_height, supply_height + 0.5))
        ax3d.set_title("3D Routing Sketch (not to scale)")
        st.pyplot(fig_3d)
        
        col_layout1, col_layout2 = st.columns(2)
        with col_layout1:
            st.metric("Main Duct", f"{main_diameter} mm", f"{main_velocity:.1f} m/s")
            st.metric("Branch Duct", f"{branch_diameter} mm", f"{branch_velocity:.1f} m/s")
        with col_layout2:
            st.metric("Diffuser Count", diffuser_count)
            st.metric("Flow per Diffuser", f"{branch_flow:.1f} l/s")
        
        report_data.update({
            "duct_shape": duct_shape,
            "duct_flow_ls": duct_flow_rate,
            "duct_main_size_mm": main_diameter if duct_shape == "Circular" else eq_circ,
            "duct_branch_size_mm": branch_diameter,
            "duct_velocity_main": main_velocity,
            "duct_velocity_branch": branch_velocity,
            "duct_pressure_drop_pa_per_m": dp_per_m,
            "duct_friction_factor": f_factor,
            "duct_reynolds": reynolds,
            "duct_diffuser_count": diffuser_count
        })
    
    # ========================================================================
    # TAB 4: PSYCHROMETRICS
    # ========================================================================
    with tab4:
        st.header("Psychrometric Calculator")
        st.markdown("*Calculate air properties for HVAC system design*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Conditions")
            
            dry_bulb = st.number_input(
                "Dry Bulb Temperature (°C)",
                value=20.0,
                min_value=-20.0,
                max_value=50.0,
                step=0.5,
                help="Air temperature measured by a standard thermometer"
            )
            
            relative_humidity = st.slider(
                "Relative Humidity (%)",
                min_value=0,
                max_value=100,
                value=50,
                help="Percentage of moisture in air relative to saturation"
            )
            
            st.markdown("---")
            
            # Calculate psychrometric properties
            calc = PsychrometricCalculator()
            
            moisture_content = calc.moisture_content(dry_bulb, relative_humidity, P_atm)
            dew_point = calc.dew_point(dry_bulb, relative_humidity)
            enthalpy = calc.enthalpy(dry_bulb, relative_humidity, P_atm)
            vapor_pressure = calc.vapor_pressure(dry_bulb, relative_humidity)
            report_data.update({
                "psychro_dry_bulb": dry_bulb,
                "psychro_rh": relative_humidity,
                "psychro_enthalpy": enthalpy,
                "psychro_dew_point": dew_point,
                "psychro_moisture": moisture_content
            })
            
            st.subheader("Calculated Properties")
            
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Moisture Content", f"{moisture_content:.2f} g/kg")
                st.metric("Enthalpy", f"{enthalpy:.2f} kJ/kg")
            
            with col1b:
                st.metric("Dew Point", f"{dew_point:.1f} °C")
                st.metric("Vapor Pressure", f"{vapor_pressure:.0f} Pa")
            
            # Comfort analysis
            st.markdown("---")
            st.subheader("Thermal Comfort Analysis")
            
            # CIBSE comfort criteria
            comfort_temp_min = 20
            comfort_temp_max = 24
            comfort_rh_min = 40
            comfort_rh_max = 70
            
            temp_ok = comfort_temp_min <= dry_bulb <= comfort_temp_max
            rh_ok = comfort_rh_min <= relative_humidity <= comfort_rh_max
            
            if temp_ok and rh_ok:
                st.markdown('<div class="insight-box">✅ <strong>Comfortable:</strong> Conditions within CIBSE comfort range</div>', 
                           unsafe_allow_html=True)
            else:
                issues = []
                if not temp_ok:
                    if dry_bulb < comfort_temp_min:
                        issues.append("Temperature too low")
                    else:
                        issues.append("Temperature too high")
                if not rh_ok:
                    if relative_humidity < comfort_rh_min:
                        issues.append("Humidity too low (dry air)")
                    else:
                        issues.append("Humidity too high (muggy)")
                
                st.markdown(f'<div class="warning-box">⚠️ <strong>Outside Comfort Zone:</strong> {", ".join(issues)}</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Thermal Comfort (PMV/PPD)")
            mrt = st.number_input("Mean Radiant Temperature (°C)", value=dry_bulb, min_value=-20.0, max_value=50.0, step=0.5, key="mrt")
            air_velocity = st.slider("Air Velocity (m/s)", min_value=0.0, max_value=1.5, value=0.1, step=0.05, key="air_vel")
            met_rate = st.number_input("Metabolic Rate (met)", value=1.2, min_value=0.8, max_value=3.0, step=0.1, key="met_rate")
            clothing = st.number_input("Clothing Level (clo)", value=0.7, min_value=0.0, max_value=2.0, step=0.1, key="clo_level")
            pmv, ppd = PsychrometricCalculator.pmv_ppd(dry_bulb, mrt, air_velocity, relative_humidity, met=met_rate, clo=clothing)
            col_pmv1, col_pmv2 = st.columns(2)
            with col_pmv1:
                st.metric("PMV", f"{pmv:.2f}")
            with col_pmv2:
                st.metric("PPD (%)", f"{ppd:.1f}%")
            report_data.update({
                "pmv": pmv,
                "ppd": ppd,
                "mrt": mrt,
                "air_velocity": air_velocity
            })
        
        with col2:
            st.subheader("Psychrometric Chart (Simplified)")
            
            # Create a simplified psychrometric chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Temperature range
            temp_range = np.linspace(-5, 40, 100)
            
            # Plot constant RH lines
            rh_lines = [20, 40, 60, 80, 100]
            for rh in rh_lines:
                moisture_values = [calc.moisture_content(t, rh, P_atm) for t in temp_range]
                linestyle = '-' if rh == 100 else '--'
                alpha = 0.8 if rh == 100 else 0.4
                ax.plot(temp_range, moisture_values, linestyle=linestyle, 
                       alpha=alpha, color='blue', linewidth=1)
                # Label
                if rh < 100:
                    ax.text(temp_range[-10], moisture_values[-10], f'{rh}%', 
                           fontsize=8, color='blue', alpha=0.6)
            
            # Plot current condition
            ax.plot(dry_bulb, moisture_content, 'ro', markersize=12, 
                   label=f'Current State\n{dry_bulb}°C, {relative_humidity}% RH', zorder=5)
            
            # Process line (cooling)
            target_temp_line = st.session_state.get("cooling_temp", 15.0)
            target_rh_line = st.session_state.get("cooling_rh", 50)
            moisture_target_line = calc.moisture_content(target_temp_line, target_rh_line, P_atm)
            ax.plot(
                [dry_bulb, target_temp_line],
                [moisture_content, moisture_target_line],
                color='red', linestyle='-.', linewidth=2, label='Cooling Process'
            )
            ax.scatter(target_temp_line, moisture_target_line, color='purple', zorder=6, label='Cooled State')
            
            # Comfort zone
            comfort_temps = [comfort_temp_min, comfort_temp_max, comfort_temp_max, comfort_temp_min, comfort_temp_min]
            comfort_moisture = [
                calc.moisture_content(comfort_temp_min, comfort_rh_min, P_atm),
                calc.moisture_content(comfort_temp_max, comfort_rh_min, P_atm),
                calc.moisture_content(comfort_temp_max, comfort_rh_max, P_atm),
                calc.moisture_content(comfort_temp_min, comfort_rh_max, P_atm),
                calc.moisture_content(comfort_temp_min, comfort_rh_min, P_atm)
            ]
            ax.fill(comfort_temps, comfort_moisture, color='green', alpha=0.2, label='CIBSE Comfort Zone')
            
            ax.set_xlabel('Dry Bulb Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Moisture Content (g/kg)', fontsize=12, fontweight='bold')
            ax.set_title('Simplified Psychrometric Chart', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', framealpha=0.9)
            ax.set_xlim([-5, 40])
            ax.set_ylim([0, 25])
            
            st.pyplot(fig)
        
        # Process Analysis
        st.markdown("---")
        st.subheader("HVAC Process Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Heating Process (Constant Moisture)**")
            target_temp_heating = st.number_input(
                "Target Temperature (°C)",
                value=25.0,
                min_value=dry_bulb,
                max_value=50.0,
                key="heating_temp"
            )
            
            # Calculate new RH after heating
            new_rh = calc.relative_humidity_from_dew_point(target_temp_heating, dew_point)
            enthalpy_after_heating = calc.enthalpy(target_temp_heating, new_rh, P_atm)
            heating_energy = enthalpy_after_heating - enthalpy
            
            st.metric("New Relative Humidity", f"{new_rh:.1f}%")
            st.metric("Heating Energy Required", f"{heating_energy:.2f} kJ/kg")
            
            # For a flow rate
            if 'total_flow_rate' in locals():
                mass_flow = (total_flow_rate / 1000) * air_density
                heating_power = mass_flow * heating_energy
                st.metric("Heating Load (from Tab 1 flow)", f"{heating_power:.2f} kW")
                report_data.update({"heating_power_kw": heating_power})

        with col4:
            st.markdown("**Cooling & Dehumidification**")
            target_temp_cooling = st.number_input(
                "Target Temperature (°C)",
                value=15.0,
                min_value=-10.0,
                max_value=dry_bulb,
                key="cooling_temp"
            )
            
            target_rh_cooling = st.slider(
                "Target RH (%)",
                min_value=30,
                max_value=70,
                value=50,
                key="cooling_rh"
            )
            
            moisture_after_cooling = calc.moisture_content(target_temp_cooling, target_rh_cooling, P_atm)
            enthalpy_after_cooling = calc.enthalpy(target_temp_cooling, target_rh_cooling, P_atm)
            cooling_energy = enthalpy - enthalpy_after_cooling
            moisture_removed = moisture_content - moisture_after_cooling
            
            st.metric("Cooling Energy Required", f"{cooling_energy:.2f} kJ/kg")
            st.metric("Moisture Removed", f"{moisture_removed:.2f} g/kg")
            
            if 'total_flow_rate' in locals():
                mass_flow = (total_flow_rate / 1000) * air_density
                cooling_power = mass_flow * cooling_energy
                condensate_rate = mass_flow * moisture_removed / 1000  # kg/s to l/h conversion
                st.metric("Cooling Load (from Tab 1 flow)", f"{cooling_power:.2f} kW")
                st.metric("Condensate Rate", f"{condensate_rate * 3.6:.2f} l/h")
                report_data.update({
                    "cooling_power_kw": cooling_power,
                    "condensate_rate_lph": condensate_rate * 3.6
                })
        
        # Engineering Insights
        st.markdown("---")
        st.subheader("💡 Engineering Insights & Recommendations")
        
        insights_psychro = []
        
        # Temperature analysis
        if dry_bulb < 18:
            insights_psychro.append("🥶 **Cold Conditions:** Heating required to maintain comfort. Consider heat recovery systems.")
        elif dry_bulb > 26:
            insights_psychro.append("🥵 **Hot Conditions:** Cooling required. High enthalpy indicates significant cooling load.")
        
        # Humidity analysis
        if relative_humidity < 30:
            insights_psychro.append("💧 **Low Humidity:** Risk of dry air complaints, static electricity, and respiratory discomfort. Consider humidification.")
        elif relative_humidity > 70:
            insights_psychro.append("💦 **High Humidity:** Risk of condensation, mold growth, and thermal discomfort. Dehumidification recommended.")
        
        # Condensation risk
        if dew_point > 15:
            insights_psychro.append(f"⚠️ **Condensation Risk:** Dew point {dew_point:.1f}°C. Any surface below this temperature will experience condensation. Insulate cold water pipes and ductwork.")
        
        # Enthalpy interpretation
        if enthalpy > 60:
            insights_psychro.append(f"⚡ **High Enthalpy:** {enthalpy:.1f} kJ/kg indicates high cooling load. Consider economizer cycle when outdoor conditions are favorable.")
        
        # Mixed air calculation tip
        insights_psychro.append("💡 **Design Tip:** For mixed air calculations, use h_mix = (m1·h1 + m2·h2)/(m1+m2) where masses are proportional to flow rates.")
        
        # Seasonal considerations
        if dry_bulb < 10:
            insights_psychro.append("❄️ **Winter Operation:** Consider heat recovery ventilation (HRV) to reduce heating energy. Typical effectiveness: 70-85%.")
        elif dry_bulb > 25 and relative_humidity > 60:
            insights_psychro.append("☀️ **Summer Operation:** High latent load. Consider enthalpy wheel or desiccant dehumidification for better humidity control.")
        
        for insight in insights_psychro:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Reference table
        st.markdown("---")
        st.subheader("📚 Quick Reference: Psychrometric Properties")
        
        ref_temps = [0, 5, 10, 15, 20, 25, 30]
        ref_rh = [30, 50, 70]
        
        ref_data = []
        for temp in ref_temps:
            for rh in ref_rh:
                w = calc.moisture_content(temp, rh, P_atm)
                h = calc.enthalpy(temp, rh, P_atm)
                td = calc.dew_point(temp, rh)
                ref_data.append({
                    "Temp (°C)": temp,
                    "RH (%)": rh,
                    "Moisture (g/kg)": f"{w:.2f}",
                    "Enthalpy (kJ/kg)": f"{h:.2f}",
                    "Dew Point (°C)": f"{td:.1f}"
                })
        
        df_ref = pd.DataFrame(ref_data)
        st.dataframe(df_ref, use_container_width=True, height=300)
    
    # ========================================================================
    # TAB 5: FAN LAWS
    # ========================================================================
    with tab5:
        st.header("Fan Laws")
        st.markdown("*Predict performance changes with speed adjustments (ASHRAE fan laws)*")
        
        col_fl1, col_fl2 = st.columns(2)
        with col_fl1:
            flow_current = st.number_input("Current Flow (l/s)", value=500.0, min_value=0.0, step=10.0, key="fan_flow")
            pressure_current = st.number_input("Current Pressure (Pa)", value=500.0, min_value=0.0, step=10.0, key="fan_pressure")
            power_current = st.number_input("Current Power (W)", value=600.0, min_value=0.0, step=10.0, key="fan_power")
            rpm_current = st.number_input("Current RPM", value=1200.0, min_value=1.0, step=50.0, key="fan_rpm")
        
        with col_fl2:
            rpm_new_direct = st.number_input("New RPM", value=1400.0, min_value=1.0, step=50.0, key="fan_new_rpm")
            speed_change_pct = st.slider("Speed Change (%)", min_value=-50, max_value=100, value=16, step=1,
                                         help="Positive increases speed, negative reduces speed",
                                         key="fan_speed_change")
            rpm_new = rpm_current * (1 + speed_change_pct / 100)
            rpm_target = rpm_new_direct if rpm_new_direct else rpm_new
            st.caption(f"Computed RPM from percent change: {rpm_new:.0f} rpm")
        
        ratio = rpm_target / rpm_current if rpm_current else 0
        new_flow = flow_current * ratio
        new_pressure = pressure_current * ratio ** 2
        new_power = power_current * ratio ** 3
        
        col_fl3, col_fl4, col_fl5 = st.columns(3)
        with col_fl3:
            st.metric("New Flow (l/s)", f"{new_flow:.1f}")
        with col_fl4:
            st.metric("New Pressure (Pa)", f"{new_pressure:.1f}")
        with col_fl5:
            st.metric("New Power (W)", f"{new_power:.1f}", help="Power changes with the cube of speed — watch energy use!")
        
        st.info("Fan Laws: Q₂/Q₁ = N₂/N₁, P₂/P₁ = (N₂/N₁)², Power₂/Power₁ = (N₂/N₁)³")
        
        report_data.update({
            "fan_law_flow": new_flow,
            "fan_law_pressure": new_pressure,
            "fan_law_power": new_power,
            "fan_law_rpm_target": rpm_target
        })
    
    # Sidebar report download (after calculations)
    report_text = generate_report(report_data)
    if generate_now:
        st.sidebar.download_button(
            "⬇️ Download Report (.txt)",
            data=report_text,
            file_name="hvac_report.txt"
        )
        st.sidebar.success("Report generated. Download ready.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>HVAC Decision Support System v1.0</strong></p>
            <p>Developed for TU Dublin Building Engineering (TU825)</p>
            <p>Compliant with CIBSE Guide A & B | Part L Building Regulations</p>
            <p style='font-size: 0.8rem; margin-top: 1rem;'>
                ⚠️ This tool provides design guidance. Always verify calculations and consult relevant standards for final design decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
