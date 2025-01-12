# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Checks if the config settings and the network is sound.
"""

import logging
from scripts._helpers import configure_logging, mock_snakemake
import pypsa

logger = logging.getLogger(__name__)

def check_basic_config_settings(n, config):
    """
    Checks if the basic settings are correct.
    """

    if config["ammonia"] != "regional":
        logger.error("Ammonia sector must be regional. Please set config['sector']['ammonia'] to 'regional'.")

    if not config["steel"]["endogenous"]:
        logger.error(
            "Endogenous steel demand must be activated. Please set config['sector']['steel']['endogenous'] to True."
        )

    if not config["H2_network"]:
        logger.error(
            "H2 network with regional demand must be activated. Please set config['sector']['H2_network'] to True."
        )

    if (
        not config["methanol"]["regional_methanol_demand"]
        or not config["regional_oil_demand"]
        or not config["regional_coal_demand"]
    ):
        logger.error(
            "Regional methanol, oil and coal demand must be activated. Please set config['sector']['methanol']['regional_methanol_demand'], config['sector']['regional_oil_demand'] and config['sector']['regional_coal_demand'] to True."
        )
    
    if config["co2network"]:
        logger.error(
            "CO2 network not working yet. Please add code to the function adjust_industry_loads()."
        )

    return

def check_meoh_architecture(n):
    if config["relocation"] == "methanol" or config["relocation"] == "all":
        # shipping methanol is connected to DE or EU methanol bus
        assert(n.links[(n.links.carrier == "shipping methanol") & (n.links.bus0.str[:2] == "DE")].bus0.unique() == "DE methanol")
        assert(n.links[(n.links.carrier == "shipping methanol") & (n.links.bus0.str[:2] == "EU")].bus0.unique() == "EU methanol")
        # industry methanol is connected to DE or EU methanol bus
        assert(n.links[(n.links.carrier == "industry methanol") & (n.links.bus0.str[:2] == "DE")].bus0.unique() == "DE methanol")
        assert(n.links[(n.links.carrier == "industry methanol") & (n.links.bus0.str[:2] == "EU")].bus0.unique() == "EU methanol")
        # there is a shipping/industry methanol load at each node
        assert(n.loads[n.loads.carrier == "shipping methanol"].shape[0] == 68)
        assert(n.loads[n.loads.carrier == "industry methanol"].shape[0] == 68)
        # there is a transport links from DE to EU and vice versa
        assert(not n.links.loc["EU methanol -> DE methanol"].empty)
        assert(not n.links.loc["DE methanol -> EU methanol"].empty)
    else:
        # each node should have a methannol bus
        assert(n.buses[n.buses.carrier == "methanol"].shape[0] == 68)
        # each node should have a methanol industry load
        assert(n.loads[n.loads.carrier == "shipping methanol"].shape[0] == 68)
        # each methanolisation plant should connect to their own methanol bus
        assert(n.links[n.links.carrier == "methanolisation"].bus0.str[:2].equals(n.links[n.links.carrier == "methanolisation"].bus1.str[:2]))
        # there is no EU methanol bus
        assert(n.buses[(n.buses.index.str.contains("EU")) & (n.buses.index.str.contains("methanol"))].empty)
        assert(n.buses[(n.buses.index.str.contains("EU")) & (n.buses.index.str.contains("meoh"))].empty)
        # there is no DE - EU methanol transpot link
        assert("EU methanol -> DE methanol" not in n.links.index)
        assert("DE methanol -> EU methanol" not in n.links.index)
    return


def check_nh3_architecture(n):
    if config["relocation"] == "ammonia" or config["relocation"] == "all":
        # Haber Bosch links are connected to DE or EU NH3 bus
        assert(n.links[(n.links.carrier == "Haber-Bosch") & (n.links.bus0.str[:2] == "DE")].bus1.unique() == "DE NH3")
        assert(n.links[(n.links.carrier == "Haber-Bosch") & (n.links.bus0.str[:2] != "DE")].bus1.unique() == "EU NH3")
        # there are two NH3 loads
        assert(not n.loads.loc["DE NH3"].empty)
        assert(not n.loads.loc["EU NH3"].empty)
        # there is a shipping/industry methanol load at each node
        assert(n.loads[n.loads.carrier == "shipping methanol"].shape[0] == 68)
        assert(n.loads[n.loads.carrier == "industry methanol"].shape[0] == 68)
        # there is a transport links from DE to EU and vice versa
        assert(not n.links.loc["EU NH3 -> DE NH3"].empty)
        assert(not n.links.loc["DE NH3 -> EU NH3"].empty)
    else:
        # each node should have a ammonia bus
        assert(n.buses[n.buses.carrier == "NH3"].shape[0] == 68)
        # each node should have a ammonia load
        assert(n.loads[n.loads.carrier == "NH3"].shape[0] == 68)
        # each Haber-Bosch plant should connect to their own NH3 bus
        assert(n.links[n.links.carrier == "Haber-Bosch"].bus0.str[:2].equals(n.links[n.links.carrier == "Haber-Bosch"].bus1.str[:2]))
        # there is no EU NH3 bus
        assert(n.buses[(n.buses.index.str.contains("EU")) & (n.buses.index.str.contains("NH3"))].empty)
        assert(n.buses[(n.buses.index.str.contains("EU")) & (n.buses.index.str.contains("ammonia"))].empty)
        # there is no DE - EU methanol transport link
        assert("EU NH3 -> DE NH3" not in n.links.index)
        assert("DE NH3 -> EU NH3" not in n.links.index)
    return


def check_steel_architecture(n):
    if config["relocation"] == "steel" or config["relocation"] == "all":
        # DRI/EAF links are connected to DE or EU hbi/steel bus
        assert(n.links[(n.links.carrier == "DRI") & (n.links.bus0.str[:2] == "DE")].bus1.unique() == "DE hbi")
        assert(n.links[(n.links.carrier == "DRI") & (n.links.bus0.str[:2] != "DE")].bus1.unique() == "EU hbi")
        assert(n.links[(n.links.carrier == "EAF") & (n.links.bus0.str[:2] == "DE")].bus1.unique() == "DE steel")
        assert(n.links[(n.links.carrier == "EAF") & (n.links.bus0.str[:2] != "DE")].bus1.unique() == "EU steel")
        # there are two steel loads
        assert(not n.loads.loc["DE steel"].empty)
        assert(not n.loads.loc["EU steel"].empty)
        # there is a transport links from DE to EU and vice versa
        assert(not n.links.loc["EU steel -> DE steel"].empty)
        assert(not n.links.loc["DE steel -> EU steel"].empty)
        assert(not n.links.loc["EU hbi -> DE hbi"].empty)
        assert(not n.links.loc["DE hbi -> EU hbi"].empty)
    else:
        # DRI/EAF links are connected to DE or EU NH3 bus
        assert(n.links[n.links.carrier == "DRI"].bus0.str[:2].equals(n.links[n.links.carrier == "DRI"].bus1.str[:2]))
        assert(n.links[n.links.carrier == "EAF"].bus0.str[:2].equals(n.links[n.links.carrier == "EAF"].bus1.str[:2]))
        assert(n.links[n.links.carrier == "DRI"].shape[0] == 68)
        assert(n.links[n.links.carrier == "EAF"].shape[0] == 68)
        # there are 68 steel loads
        assert(n.loads[n.loads.carrier == "steel"].shape[0] == 68)
        # there is no EU steel/hbi bus
        assert(n.buses[(n.buses.index.str.contains("EU")) & (n.buses.index.str.contains("steel"))].empty)
        assert(n.buses[(n.buses.index.str.contains("EU")) & (n.buses.index.str.contains("hbi"))].empty)
        # there is no DE - EU steel/hbi transport link
        assert("EU steel -> DE steel" not in n.links.index)
        assert("DE steel -> EU steel" not in n.links.index)
        assert("EU hbi -> DE hbi" not in n.links.index)
        assert("DE hbi -> EU hbi" not in n.links.index)



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "modify_final_network",
            simpl="",
            clusters=68,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2030",
            run="eu_import-nh3_relocation",
        )

    configure_logging(snakemake)
    logger.info("Checking the prenetwork for consistency.")

    config = snakemake.params.config
    n = pypsa.Network(snakemake.input.network)

    check_basic_config_settings(n, config)

    check_meoh_architecture(n)

    check_nh3_architecture(n)

    check_steel_architecture(n)

    logger.info("All checks passed.")