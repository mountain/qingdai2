from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColumnStepOut:
    Ta: np.ndarray
    P_cond: np.ndarray
    q_next: np.ndarray
    RH: np.ndarray
    cloud_eff: np.ndarray


def resolve_total_seconds(
    *,
    dt: float,
    day_in_seconds: float,
    orbital_period: float,
    duration_days: float | None,
    n_steps: int | None,
    total_years: float | None,
    sim_days: float | None,
    default_orbit_fraction: float,
) -> float:
    if duration_days is not None:
        return float(duration_days) * day_in_seconds
    if n_steps is not None:
        return float(n_steps) * dt
    if total_years is not None:
        return float(total_years) * orbital_period
    if sim_days is not None:
        return float(sim_days) * day_in_seconds
    return float(default_orbit_fraction) * orbital_period


def compute_column_step(
    *,
    h_read: np.ndarray,
    q_read: np.ndarray,
    cloud_read: np.ndarray,
    dt: float,
    hum_mod,
    hum_params,
) -> ColumnStepOut:
    Ta = 288.0 + (9.81 / 1004.0) * h_read
    P_cond, q_next = hum_mod.condensation(q_read, Ta, dt, hum_params)
    qsat = hum_mod.q_sat(Ta, p=hum_params.p0)
    RH = np.clip(q_next / np.maximum(qsat, 1.0e-9), 0.0, 1.2)
    cloud_eff = np.clip(cloud_read, 0.0, 1.0)
    return ColumnStepOut(Ta=Ta, P_cond=P_cond, q_next=q_next, RH=RH, cloud_eff=cloud_eff)


def blend_ecology_albedo(
    *,
    base_albedo: np.ndarray,
    land_mask: np.ndarray,
    eco,
    eco_weight: float,
    eco_albedo_couple: bool,
    insolation: np.ndarray,
    cloud_eff: np.ndarray,
    dt: float,
) -> np.ndarray:
    albedo = base_albedo
    if eco is None or (not eco_albedo_couple):
        return albedo
    alpha_eco = eco.step_subdaily(insolation, cloud_eff, dt)
    if alpha_eco is None:
        return albedo
    alpha_mix = np.where(np.isfinite(alpha_eco), alpha_eco, albedo)
    return np.where(
        land_mask == 1,
        np.clip((1.0 - eco_weight) * albedo + eco_weight * alpha_mix, 0.0, 1.0),
        albedo,
    )


def normalize_fluxes(fluxes, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fluxes is None:
        zeros = np.zeros_like(ref)
        return zeros, zeros, zeros
    evap_flux = (
        np.zeros_like(ref)
        if fluxes.evap_flux is None
        else np.maximum(0.0, np.asarray(fluxes.evap_flux))
    )
    Q_net = np.where(np.isfinite(np.asarray(fluxes.Qnet)), np.asarray(fluxes.Qnet), 0.0)
    precip_flux = (
        np.zeros_like(ref)
        if fluxes.precip_flux is None
        else np.maximum(0.0, np.asarray(fluxes.precip_flux))
    )
    return evap_flux, Q_net, precip_flux


def compute_energy_diag(
    *,
    energy_mod,
    eparams,
    insolation: np.ndarray,
    albedo: np.ndarray,
    cloud_eff: np.ndarray,
    Ts_read: np.ndarray,
    Ta: np.ndarray,
    land_mask: np.ndarray,
    ice_mask: np.ndarray,
    u_read: np.ndarray,
    v_read: np.ndarray,
    evap_flux: np.ndarray,
    lat_mesh: np.ndarray,
    ch: float,
    cp_air: float,
    bowen_land: float,
    bowen_ocean: float,
    latent_heat: float,
) -> dict:
    sw_atm, sw_sfc, r_diag = energy_mod.shortwave_radiation(insolation, albedo, cloud_eff, eparams)
    eps_sfc_map = energy_mod.surface_emissivity_map(land_mask, np.where(ice_mask, 1.0, 0.0))
    _lw_atm, lw_sfc, olr_diag, _dlr, _eps = energy_mod.longwave_radiation_v2(
        Ts_read, Ta, cloud_eff, eps_sfc_map, eparams
    )
    sh_arr, _ = energy_mod.boundary_layer_fluxes(
        Ts_read,
        Ta,
        u_read,
        v_read,
        land_mask,
        C_H=ch,
        rho=1.2,
        c_p=cp_air,
        B_land=bowen_land,
        B_ocean=bowen_ocean,
    )
    lh_arr = np.where(np.isfinite(evap_flux), latent_heat * evap_flux, 0.0)
    return energy_mod.compute_energy_diagnostics(
        lat_mesh,
        insolation,
        r_diag,
        olr_diag,
        sw_sfc,
        lw_sfc,
        sh_arr,
        lh_arr,
    )
