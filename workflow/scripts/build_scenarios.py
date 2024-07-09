# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script reads in data from the IIASA database to create the scenario.yaml file
import logging
logger = logging.getLogger(__name__)

import ruamel.yaml
from pathlib import Path
import pandas as pd
import os

def get_transport_growth(df, planning_horizons):
    # Aviation growth factor - using REMIND-EU v1.1 since Aladin v1 does not include bunkers
    aviation_model = "REMIND-EU v1.1"
    aviation = df.loc[aviation_model,"Final Energy|Bunkers|Aviation", "PJ/yr"]
    aviation_growth_factor = aviation / aviation[2020]

    return aviation_growth_factor[planning_horizons]


def get_primary_steel_share(df, planning_horizons):
    # Get share of primary steel production
    model = "FORECAST v1.0"
    total_steel = df.loc[model, "Production|Steel"]
    primary_steel = df.loc[model, "Production|Steel|Primary"]
    
    primary_steel_share = primary_steel / total_steel
    primary_steel_share = primary_steel_share[planning_horizons]

    if model == "FORECAST v1.0" and planning_horizons[0] == 2020:
        logger.warning("FORECAST v1.0 does not have data for 2020. Using 2021 data for Production|Steel instead.")
        primary_steel_share[2020] = primary_steel[2021] / total_steel[2021]

    
    return primary_steel_share.set_index(pd.Index(["Primary_Steel_Share"]))

def get_DRI_share(df, planning_horizons):
    # Get share of DRI steel production
    model = "FORECAST v1.0"
    total_steel = df.loc[model, "Production|Steel|Primary"]
    # Assuming that only hydrogen DRI steel is sustainable and DRI using natural gas is phased out
    DRI_steel = df.loc[model, "Production|Steel|Primary|Direct Reduction Hydrogen"]

    DRI_steel_share = DRI_steel / total_steel

    if model == "FORECAST v1.0" and planning_horizons[0] == 2020:
        logger.warning("FORECAST v1.0 does not have data for 2020. Using 2021 data for DRI fraction instead.")
        DRI_steel_share[2020] = DRI_steel_share[2021] / total_steel[2021]
    
    DRI_steel_share = DRI_steel_share[planning_horizons]

    return DRI_steel_share.set_index(pd.Index(["DRI_Steel_Share"]))


def get_steam_share(df):
    # Get share of steam production from FORECAST v1.0
    model = "FORECAST v1.0"
    
    biomass = df.loc[model, "Final Energy|Industry excl Non-Energy Use|Solids|Biomass"][2045]
    hydrogen = df.loc[model, "Final Energy|Industry excl Non-Energy Use|Hydrogen"][2045]
    electricity = df.loc[model, "Final Energy|Industry excl Non-Energy Use|Electricity"][2045]
    total = biomass + hydrogen + electricity

    biomass_fraction = biomass / total
    hydrogen_fraction = hydrogen / total
    electricity_fraction = electricity / total

    return biomass_fraction.iloc[0], hydrogen_fraction.iloc[0], electricity_fraction.iloc[0]


def get_co2_budget(df, source):
    # relative to the DE emissions in 1990 *including bunkers*; also
    # account for non-CO2 GHG and allow extra room for international
    # bunkers which are excluded from the national targets

    # Baseline emission in DE in 1990 in Mt as understood by the KSG and by PyPSA
    baseline_co2 = 1251
    baseline_pypsa = 1052
    if source == "KSG":
        ## GHG targets according to KSG
        initial_years_co2 = pd.Series(
            index = [2020, 2025, 2030],
            data = [813, 643, 438],
        )

        later_years_co2 = pd.Series(
            index = [2035, 2040, 2045, 2050],
            data = [0.77, 0.88, 1.0, 1.0],
        )

        targets_co2 = pd.concat(
            [initial_years_co2, (1 - later_years_co2) * baseline_co2],
        )
    elif source == "UBA":
        ## For Zielverfehlungsszenarien use UBA Projektionsbericht
        targets_co2 = pd.Series(
            index = [2020, 2025, 2030, 2035, 2040, 2045, 2050],
            data = [813, 655, 455, 309, 210, 169, 157],
        )
    else:
        raise ValueError("Invalid source for CO2 budget.")
    ## Compute nonco2 from Ariadne-Leitmodell (REMIND)

    co2 = (
        df.loc["Emissions|CO2 incl Bunkers","Mt CO2/yr"]  
        - df.loc["Emissions|CO2|Land-Use Change","Mt CO2-equiv/yr"]
        - df.loc["Emissions|CO2|Energy|Demand|Bunkers","Mt CO2/yr"]
    )

    ghg = (
        df.loc["Emissions|Kyoto Gases","Mt CO2-equiv/yr"]
        - df.loc["Emissions|Kyoto Gases|Land-Use Change","Mt CO2-equiv/yr"]
        # No Kyoto Gas emissions for Bunkers recorded in Ariadne DB
    )

    nonco2 = ghg - co2

    ## PyPSA disregards nonco2 GHG emissions, but includes bunkers

    targets_pypsa = (
        targets_co2 - nonco2 
        + df.loc["Emissions|CO2|Energy|Demand|Bunkers","Mt CO2/yr"]
    )

    target_fractions_pypsa = (
        targets_pypsa.loc[targets_co2.index] / baseline_pypsa
    )

    return target_fractions_pypsa.round(3)


def write_to_scenario_yaml(
        input, output, scenarios, df):
    # read in yaml file
    yaml = ruamel.yaml.YAML()
    file_path = Path(input)
    config = yaml.load(file_path)
    for scenario in scenarios:
        reference_scenario = config[scenario]["iiasa_database"]["reference_scenario"]
        fallback_reference_scenario = config[scenario]["iiasa_database"]["fallback_reference_scenario"]
        if reference_scenario == "KN2045plus_EasyRide":
            fallback_reference_scenario = reference_scenario
        co2_budget_source = config[scenario]["co2_budget_DE_source"]

        co2_budget_fractions = get_co2_budget(
            df.loc["REMIND-EU v1.1", fallback_reference_scenario],
            co2_budget_source
        )
        
        planning_horizons = [2020, 2025, 2030, 2035, 2040, 2045] # for 2050 we still need data
        
        aviation_demand_factor = get_transport_growth(df.loc[:, fallback_reference_scenario, :], planning_horizons)

        config[scenario]["sector"] = {}
        
        config[scenario]["sector"]["aviation_demand_factor"] = {}
        for year in planning_horizons:
            config[scenario]["sector"]["aviation_demand_factor"][year] = round(aviation_demand_factor.loc[year].item(), 4)

        st_primary_fraction = get_primary_steel_share(df.loc[:, reference_scenario, :], planning_horizons)
        
        dri_fraction = get_DRI_share(df.loc[:, reference_scenario, :], planning_horizons)

        config[scenario]["industry"] = {}
        config[scenario]["industry"]["St_primary_fraction"] = {}
        config[scenario]["industry"]["DRI_fraction"] = {}
        for year in st_primary_fraction.columns:
            config[scenario]["industry"]["St_primary_fraction"][year] = round(st_primary_fraction.loc["Primary_Steel_Share", year].item(), 4)
            config[scenario]["industry"]["DRI_fraction"][year] = round(dri_fraction.loc["DRI_Steel_Share", year].item(), 4)

        biomass_share, hydrogen_share, electricity_share = get_steam_share(df.loc[:, reference_scenario, :])

        config[scenario]["industry"]["steam_biomass_fraction"] = {}
        config[scenario]["industry"]["steam_hydrogen_fraction"] = {}
        config[scenario]["industry"]["steam_electricity_fraction"] = {}
        config[scenario]["industry"]["steam_biomass_fraction"] = float(round(biomass_share, 4))
        config[scenario]["industry"]["steam_hydrogen_fraction"] = float(round(hydrogen_share, 4))
        config[scenario]["industry"]["steam_electricity_fraction"] = float(round(electricity_share, 4))

        config[scenario]["co2_budget_national"] = {}
        for year, target in co2_budget_fractions.items():
            config[scenario]["co2_budget_national"][year] = {}
            config[scenario]["co2_budget_national"][year]["DE"] = target

    # write back to yaml file
    yaml.dump(config, Path(output))



if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_scenarios")

    # Set USERNAME and PASSWORD for the Ariadne DB
    ariadne_db = pd.read_csv(
        snakemake.input.ariadne_database,
        index_col=["model", "scenario", "region", "variable", "unit"]
    )
    ariadne_db.columns = ariadne_db.columns.astype(int)

    df = ariadne_db.loc[
        :, 
        :,
        "Deutschland"]
    
    scenarios = snakemake.params.scenarios

    input = snakemake.input.scenario_yaml
    output = snakemake.output.scenario_yaml

    # for scenario in scenarios:
    write_to_scenario_yaml(
        input, output, scenarios, df)
