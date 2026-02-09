from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from contextlib import asynccontextmanager
from datetime import datetime
    
class SensorData(BaseModel):
    timestamp: datetime
    load_pct: float
    coal_flow_tph: float
    sh_press_mpa: float
    sh_temp_c: float
    rh_temp_c: float
    drum_level_mm: float
    furnace_draft_pa: float
    flue_o2_pct: float
    flue_temp_c: float
    feedwater_flow_tph: float
    hp_inlet_press_mpa: float
    hp_exhaust_press_mpa: float
    ip_exhaust_press_mpa: float
    lp_exhaust_press_kpa: float
    turbine_speed_rpm: float
    vib_hp_um: float
    vib_ip_um: float
    vib_lp_um: float
    bearing_temp1_c: float
    bearing_temp2_c: float
    condenser_vacuum_kpa_abs: float
    hotwell_level_m: float
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_flow_m3h: float
    gen_mw: float
    gen_mvar: float
    terminal_voltage_kv: float
    frequency_hz: float
    power_factor: float
    stator_temp_c: float
    rotor_temp_c: float
    nox_ppm: float
    sox_ppm: float
    co_ppm: float
    dust_mg_nm3: float
    bottom_ash_level_pct: float
    fly_ash_hopper_level_pct: float
    fd_fan_amp_a: float
    id_fan_amp_a: float
    mill_current_a: float
    pa_fan_amp_a: float
    boiler_efficiency_pct: float
    plant_efficiency_pct: float
    
    class Config:
        extra = "allow"
        
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['watchdog'] = joblib.load('watchdog_model.pkl')
    ml_model['localizer'] = joblib.load('localizer_model.pkl')
    ml_model['scaler'] = joblib.load('scaler.pkl')
    ml_model['sigma'] = joblib.load('sigma_stats.pkl')
    ml_model['feature_cols'] = ml_model['scaler'].feature_names_in_
    print("--------------ML Models and Artifacts Loaded Successfully--------------")
    yield
    ml_model.clear()
    
app = FastAPI(lifespan=lifespan)


def get_equipment_context(sensor_names):
    """
    Maps sensor names to their respective equipment subsystems based on predefined keyword associations.
    """
    
    subsystems = {
        'Boiler': ['coal', 'steam', 'drum', 'flue', 'feedwater', 'furnace', 'boiler', 'sh_', 'rh_'],
        'Turbine': ['vib_', 'bearing_', 'turbine', 'speed', 'hp_', 'ip_', 'lp_'],
        'Condenser': ['condenser', 'cw_', 'hotwell', 'vacuum'],
        'Generator': ['gen_', 'frequency', 'voltage', 'power_factor', 'stator', 'rotor'],
        'Emissions': ['nox', 'sox', 'co_', 'dust', 'ash']
    }
    
    systems = []
    system_anomaly_count = {}
    for sensor in sensor_names:
            for system, keywords in subsystems.items():
                if any(k in sensor for k in keywords):
                     #count the number of anomalies per system
                     system_anomaly_count[system] = system_anomaly_count.get(system, 0) + 1
                    
                    ##change it to append system if already not present
                     if system not in systems:
                        systems.append(system)
    if systems:
         return (systems, system_anomaly_count)
    return "Plant Auxiliary" 

@app.post("/analyze")
async def analyze_plant_data(data: SensorData):
    try:
        input_dict = data.model_dump()
        
        df_input = pd.DataFrame([input_dict])
        
        #Preprocessing
        try:
            feature_needed = ml_model['feature_cols']
            df_features = df_input[feature_needed]
        except KeyError as e:
            return JSONResponse(content={"status":"error", "message": f"Missing sensor data: str{e}"})
        
        X_scaled = ml_model["scaler"].transform(df_features)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_needed)
        
        #Watchdog Layer
        watchdog_input = X_scaled_df[['load_pct', 'cw_inlet_temp_c', 'frequency_hz']]
        expected_eff = ml_model['watchdog'].predict(watchdog_input)[0]
        actual_eff = input_dict['plant_efficiency_pct']
        
        #Calculating the gap
        eff_loss = float(expected_eff - actual_eff)
        
        #Check Threshold
        if eff_loss < 1.5:
            return JSONResponse(content={"status":"Healthy","message": f"Efficieny Gap is minimal ({eff_loss:.2f}%)"}, status_code=201)
        
        reconstruced = ml_model['localizer'].predict(X_scaled_df)
        error = np.power(X_scaled_df-reconstruced, 2).values[0]
        
        suspects_df = pd.DataFrame({
            'Features': feature_needed,
            'Error_Score': error,
            'Actual_Value': df_features.values[0],
            'Expected_Value': ml_model["scaler"].inverse_transform(reconstruced)[0]
            })
        
        #Filter out efficiency itself and sort the error
        suspects_df = suspects_df[suspects_df['Features'] != 'plant_efficiency_pct']
        top_suspects = suspects_df.sort_values(by='Error_Score', ascending=False).head(5)
        print(top_suspects)
        
        #Construct JSON Dictionaries
        
        anomalous_params = {}
        normal_ranges = {}
        
        spc_status_msg = "Contextual Anomaly (Physics Mismatch)"
        
        print("Calculating Anomalous Parameters and Normal Ranges:")
        for _, row in top_suspects.iterrows():
            sensor = row['Features']
            actual = float(row['Actual_Value'])
            expected = float(row['Expected_Value'])
            
            anomalous_params[sensor] = round(actual, 2)
            
            # RETRIEVE THE SIGMA FOR THIS SPECIFIC SENSOR
            # We handle cases where the sensor might not be in the sigma list (unlikely)
            sigma = ml_model['sigma'].get(sensor, 1.0) 
            
            # CALCULATE DYNAMIC 3-SIGMA LIMITS
            # 99.7% of "Normal" data falls within +/- 3 Sigma
            lower_limit = round(expected - (3 * sigma), 2)
            upper_limit = round(expected + (3 * sigma), 2)

            # Check if "Out of Bounds" (Hard Fault) vs "Inside Bounds" (Soft Fault)
            is_hard_fault = (actual < lower_limit) or (actual > upper_limit)
            
            if is_hard_fault:
                range_str = f"{lower_limit} - {upper_limit}"
                spc_status_msg = "Critical Limit Crossed (Statistical Outlier)"
            else:
                range_str = f"{lower_limit} - {upper_limit}"
            normal_ranges[sensor] = range_str
            
        # 5. CONSTRUCT STRICT SCHEMA
        print("Constructing Payload with the following details:")
        payload = {
            "timestamp": str(input_dict['timestamp']),
                    
            "impact_metrics": {
                "efficiency_loss_pct": round(float(eff_loss), 2),
                "gen_mw": round(float(input_dict['gen_mw']), 2)
            },
            "operating_load_pct": round(float(input_dict['load_pct']), 2),
            "suspected_subsystems": get_equipment_context(top_suspects['Features'].values)[0],
            "subsystem_anomaly_counts": get_equipment_context(top_suspects['Features'].values)[1],
            
            "anomalous_parameters": anomalous_params,
            
            "normal_ranges": normal_ranges,

            
            "anomaly_metadata": {
                "anomaly_score": round(float(top_suspects['Error_Score'].max()), 2),
                "spc_status": spc_status_msg
            }
        }
        print("Generated Payload:")
        print(payload)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        