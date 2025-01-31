# -*- coding: utf-8 -*-
import logging
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from shapely.geometry import Point

from scripts._helpers import configure_logging, mock_snakemake, get
from scripts.prepare_sector_network import prepare_costs

logger = logging.getLogger(__name__)


DISTANCE_CRS = 3857
carriers_eleh2 = [
    "pipeline-h2",
    "shipping-lh2",
    "hvdc-to-elec"
]

carriers_all = [
    "pipeline-h2",
    "shipping-lh2",
    "shipping-lch4",
    "shipping-lnh3",
    "shipping-ftfuel",
    "shipping-meoh",
    "hvdc-to-elec",
    "shipping-hbi",
    "shipping-steel",
]

x = 10.5
y = 51.2
# TONI TODO: all done?
lng_dictionary = {
    "T0557": "GBMLF",  # South Hook LNG terminal, UK
    "T0492": "BEZEE",  # Gate LNG terminal, Netherlands
    "T0498": "PLSWI",  # Swinoujscie LNG terminal, Poland
    "T0462": "GREEV",  # Revithoussa LNG terminal, Greece
    "T0466": "ITVCE",  # Adriatic LNG terminal, Italy
    "T0522": "ESLCG",  # Barcelona LNG terminal, Spain
    "T0500": "ESLCG",  # Sines LNG terminal, Portugal
}


def add_endogenous_hvdc_import_options(n, cost_factor=1.0):
    logger.info("Add import options: endogenous hvdc-to-elec")

    cf = snakemake.params.import_options.get("endogenous_hvdc_import", {})

    if not cf["enable"]:
        return

    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    p_max_pu = xr.open_dataset(snakemake.input.hvdc_data).p_max_pu
    p_max_pu = p_max_pu.isel(importer=p_max_pu.notnull().argmax("importer"))

    p_nom_max = xr.open_dataset(snakemake.input.hvdc_data).p_nom_max
    p_nom_max = p_nom_max.isel(
        importer=p_nom_max.notnull().argmax("importer")
    ).to_pandas()
    country_shapes = gpd.read_file(snakemake.input.country_shapes)
    exporters_iso2 = [e.split("-")[0] for e in cf["exporters"]]  # noqa
    exporters = (
        country_shapes.set_index("ISO_A2").loc[exporters_iso2].representative_point()
    )
    exporters.index = cf["exporters"]

    import_links = {}
    a = regions.representative_point().to_crs(DISTANCE_CRS)

    # Prohibit routes through Russia or Belarus
    forbidden_hvdc_importers = ["FI", "LV", "LT", "EE"]
    a = a.loc[~a.index.str[:2].isin(forbidden_hvdc_importers)]

    for ct in exporters.index:
        b = exporters.to_crs(DISTANCE_CRS)[ct]
        d = a.distance(b)
        import_links[ct] = (
            d.where(d < d.quantile(cf["distance_threshold"])).div(1e3).dropna()
        )  # km
    import_links = pd.concat(import_links)
    import_links.loc[
        import_links.index.get_level_values(0).str.contains("KZ|CN|MN|UZ")
    ] *= (
        1.2  # proxy for detour through Caucasus in addition to crow-fly distance factor
    )

    with warnings.catch_warnings():
        # TONI TODO: catch warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # xlinks
        xlinks = {}
        for bus0, links in cf["xlinks"].items():
            for link in links:
                landing_point = gpd.GeoSeries(
                    [Point(link["x"], link["y"])], crs=4326
                ).to_crs(DISTANCE_CRS)
                bus1 = (
                    regions.to_crs(DISTANCE_CRS)
                    .geometry.distance(landing_point[0])
                    .idxmin()
                )
                xlinks[(bus0, bus1)] = link["length"]

    # I have each exporter to all countries with length of HVDC cable
    import_links = pd.concat([import_links, pd.Series(xlinks)], axis=0)
    import_links = import_links.drop_duplicates(keep="first")
    duplicated = import_links.index.duplicated(keep="first")
    import_links = import_links.loc[~duplicated]

    hvdc_cost = (
        import_links.values * cf["length_factor"] * costs.at["HVDC submarine", "fixed"]
        + costs.at["HVDC inverter pair", "fixed"]
    )

    buses_i = exporters.index

    n.add("Bus", buses_i, x=exporters.x, y=exporters.y)

    efficiency = cf["efficiency_static"] * cf["efficiency_per_1000km"] ** (
        import_links.values / 1e3
    )
    n.add("Carrier", "import hvdc-to-elec")
    n.add(
        "Link",
        ["import hvdc-to-elec " + " ".join(idx).strip() for idx in import_links.index],
        bus0=import_links.index.get_level_values(0),
        bus1=import_links.index.get_level_values(1),
        carrier="import hvdc-to-elec",
        p_nom_extendable=True,
        length=import_links.values,
        capital_cost=hvdc_cost * cost_factor,
        efficiency=efficiency,
        p_nom_max=+cf["p_nom_max"],
        bus2=import_links.index.get_level_values(0) + " export",
        efficiency2=-efficiency,
    )

    hours = int(snakemake.params.temporal_clustering[:-1])

    for tech in ["solar-utility", "onwind", "offwind"]:
        p_max_pu_tech = p_max_pu.sel(technology=tech).to_pandas().dropna().T
        # build average over every three lines but keep index
        p_max_pu_tech = p_max_pu_tech.resample(f"{hours}h").mean()
        exporters_tech_i = exporters.index.intersection(p_max_pu_tech.columns)

        grid_connection = costs.at["electricity grid connection", "fixed"]
        n.add("Carrier", "external " + tech)
        n.add(
            "Generator",
            exporters_tech_i,
            suffix=" " + tech,
            bus=exporters_tech_i,
            carrier="external " + tech,
            p_nom_extendable=True,
            capital_cost=(costs.at[tech, "fixed"] + grid_connection) * cost_factor,
            lifetime=costs.at[tech, "lifetime"],
            p_max_pu=p_max_pu_tech.reindex(columns=exporters_tech_i).values,
            p_nom_max=p_nom_max[tech].reindex(index=exporters_tech_i).values
            * cf["share_of_p_nom_max_available"],
        )

    # hydrogen storage
    n.add("Carrier", "external H2")
    h2_buses_i = n.add(
        "Bus", buses_i, suffix=" H2", carrier="external H2", location=buses_i
    )

    n.add(
        "Store",
        h2_buses_i,
        bus=h2_buses_i,
        carrier="external H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at[
            "hydrogen storage tank type 1 including compressor", "fixed"
        ]
        * cost_factor,
    )
    n.add("Carrier", "external H2 Electrolysis")

    n.add(
        "Link",
        h2_buses_i + " Electrolysis",
        bus0=buses_i,
        bus1=h2_buses_i,
        carrier="external H2 Electrolysis",
        p_nom_extendable=True,
        efficiency=costs.at["electrolysis", "efficiency"],
        capital_cost=costs.at["electrolysis", "fixed"] * cost_factor,
        lifetime=costs.at["electrolysis", "lifetime"],
    )
    n.add("Carrier", "external H2 Turbine")
    n.add(
        "Link",
        h2_buses_i + " H2 Turbine",
        bus0=h2_buses_i,
        bus1=buses_i,
        carrier="external H2 Turbine",
        p_nom_extendable=True,
        efficiency=costs.at["OCGT", "efficiency"],
        capital_cost=costs.at["OCGT", "fixed"]
        * costs.at["OCGT", "efficiency"]
        * cost_factor,
        lifetime=costs.at["OCGT", "lifetime"],
    )

    # battery storage
    n.add("Carrier", "external battery")
    b_buses_i = n.add(
        "Bus", buses_i, suffix=" battery", carrier="external battery", location=buses_i
    )

    n.add(
        "Store",
        b_buses_i,
        bus=b_buses_i,
        carrier="external battery",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=costs.at["battery storage", "fixed"] * cost_factor,
        lifetime=costs.at["battery storage", "lifetime"],
    )
    n.add("Carrier", "external battery charger")
    n.add(
        "Link",
        b_buses_i + " charger",
        bus0=buses_i,
        bus1=b_buses_i,
        carrier="external battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["battery inverter", "fixed"] * cost_factor,
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )
    n.add("Carrier", "external battery discharger")
    n.add(
        "Link",
        b_buses_i + " discharger",
        bus0=b_buses_i,
        bus1=buses_i,
        carrier="external battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    # add extra HVDC connections between MENA countries
    n.add("Carrier", "external HVDC")
    for bus0_bus1 in cf.get("extra_connections", []):
        bus0, bus1 = bus0_bus1.split("-")

        a = exporters.to_crs(DISTANCE_CRS).loc[bus0]
        b = exporters.to_crs(DISTANCE_CRS).loc[bus1]
        d = a.distance(b) / 1e3  # km

        capital_cost = (
            d * cf["length_factor"] * costs.at["HVDC overhead", "fixed"]
            + costs.at["HVDC inverter pair", "fixed"]
        )

        n.add(
            "Link",
            f"external HVDC {bus0_bus1}",
            bus0=bus0,
            bus1=bus1,
            carrier="external HVDC",
            p_min_pu=-1,
            p_nom_extendable=True,
            capital_cost=capital_cost * cost_factor,
            length=d,
        )


def add_import_options(
    n,
    capacity_boost=3.0,
    import_carriers=[
        "hvdc-to-elec",
        "pipeline-h2",
        "shipping-lh2",
        "shipping-lch4",
        "shipping-meoh",
        "shipping-ftfuel",
        "shipping-lnh3",
        "shipping-steel",
        "shipping-hbi",
    ],
    endogenous_hvdc=False,
):

    logger.info("Add import options: " + " ".join(import_carriers.keys()))
    fn = snakemake.input.gas_input_nodes_simplified
    import_nodes = pd.read_csv(fn, index_col=0)
    import_nodes["hvdc-to-elec"] = 15000

    import_config = snakemake.params.import_options
    cost_year = int(snakemake.wildcards.planning_horizons)
    exporters = import_config["exporters"]  # noqa: F841

    ports = pd.read_csv(snakemake.input.import_ports, index_col=0)

    translate = {
        "pipeline-h2": "pipeline",
        "hvdc-to-elec": "hvdc-to-elec",
        "shipping-lh2": "lng",
        "shipping-lch4": "lng",
        "shipping-lnh3": "lng",
    }

    bus_suffix = {
        "pipeline-h2": " H2",
        "hvdc-to-elec": "",
        "shipping-lh2": " H2",
        "shipping-lch4": " renewable gas",
        "shipping-lnh3": " NH3",
        "shipping-ftfuel": " renewable oil",
        "shipping-meoh": " methanol",
        "shipping-steel": " steel",
        "shipping-hbi": " hbi",
    }

    co2_intensity = {
        "shipping-lch4": ("gas", "CO2 intensity"),
        "shipping-ftfuel": ("oil", "CO2 intensity"),
        "shipping-meoh": ("methanolisation", "carbondioxide-input"),
    }

    terminal_capital_cost = {
        "shipping-lch4": 7018,  # €/MW/a
        "shipping-lh2": 7018 * 1.2,  # +20% compared to LNG
    }

    trace_scenario = snakemake.params.trace_scenario
    import_costs = pd.read_csv(
        snakemake.input.import_costs, delimiter=";", keep_default_na=False
    ).query(
        "year == @cost_year and scenario == @trace_scenario and exporter in @exporters"
    )
    import_costs["importer"] = import_costs["importer"].replace(lng_dictionary)

    cols = ["esc", "exporter", "importer", "value"]
    fields = ["Cost per MWh delivered", "Cost per t delivered"]  # noqa: F841
    import_costs = import_costs.query("subcategory in @fields")[cols]
    import_costs.rename(columns={"value": "marginal_cost"}, inplace=True)

    for k, v in translate.items():
        import_nodes[k] = import_nodes[v]
        ports[k] = ports.get(v)

    # XX export countries specified in config
    export_buses = (
        import_costs.query("esc in @import_carriers").exporter.unique() + " export"
    )
    # add buses and a store with the capacity that can be imported from each exporter
    n.add("Carrier", "export")
    n.add("Bus", export_buses, carrier="export")
    n.add(
        "Store",
        export_buses + " budget",
        bus=export_buses,
        carrier="export",
        e_nom=import_config["exporter_energy_limit"],
        e_initial=import_config["exporter_energy_limit"],
    )

    if endogenous_hvdc and "hvdc-to-elec" in import_carriers:
        cost_factor = import_carriers.pop("hvdc-to-elec")
        # deletes hvdc-to-elec from import_options
        add_endogenous_hvdc_import_options(n, cost_factor)

    regionalised_options = {
        "hvdc-to-elec",
        "pipeline-h2",
        "shipping-lh2",
        "shipping-lch4",
    }

    for tech in set(import_carriers).intersection(regionalised_options):

        import_nodes_tech = import_nodes[tech].dropna()
        forbidden_importers = []
        if "pipeline" in tech:
            forbidden_importers.extend(["DE", "BE", "FR", "GB"])  # internal entrypoints
            forbidden_importers.extend(
                ["EE", "LT", "LV", "FI"]
            )  # entrypoints via RU_BY
            sel = ~import_nodes_tech.index.str.contains("|".join(forbidden_importers))
            import_nodes_tech = import_nodes_tech.loc[sel]

        groupers = ["exporter", "importer"]
        df = (
            import_costs.query("esc == @tech")
            .groupby(groupers)
            .marginal_cost.min()
            .mul(import_carriers[tech])
            .reset_index()
        )

        bus_ports = ports[tech].dropna()

        df["importer"] = df["importer"].map(bus_ports.groupby(bus_ports).groups)
        df = df.explode("importer").query("importer in @import_nodes_tech.index")
        df["p_nom"] = df["importer"].map(import_nodes_tech)

        suffix = bus_suffix[tech]

        import_buses = df.importer.unique() + " import " + tech
        if tech == "shipping-lch4":
            data = import_buses.astype(str)
            domestic_buses = np.where(
                np.char.find(data, "DE") != -1, "DE renewable gas", "EU renewable gas"
            )
        else:
            domestic_buses = df.importer.unique() + suffix

        # pipeline imports require high minimum loading
        if "pipeline" in tech:
            p_min_pu = import_config["min_part_load_pipeline_imports"]
        elif "shipping" in tech:
            p_min_pu = import_config["min_part_load_shipping_imports"]
        else:
            p_min_pu = 0

        upper_p_nom_max = import_config["p_nom_max"].get(tech, np.inf)
        import_nodes_p_nom = import_nodes_tech.loc[df.importer.unique()]
        p_nom_max = (
            import_nodes_p_nom.mul(capacity_boost).clip(upper=upper_p_nom_max).values
        )
        p_nom_min = (
            import_nodes_p_nom.clip(upper=upper_p_nom_max).values
            if tech == "shipping-lch4"
            else 0
        )

        bus2 = "co2 atmosphere" if tech in co2_intensity else ""
        efficiency2 = (
            -costs.at[co2_intensity[tech][0], co2_intensity[tech][1]]
            if tech in co2_intensity
            else np.nan
        )

        n.add("Carrier", "import " + tech)
        n.add("Carrier", "import infrastructure " + tech)
        n.add("Bus", import_buses, carrier="import " + tech)

        n.add(
            "Link",
            pd.Index(df.exporter + " " + df.importer + " import " + tech),
            bus0=df.exporter.values + " export",
            bus1=df.importer.values + " import " + tech,
            carrier="import " + tech,
            marginal_cost=df.marginal_cost.values,
            p_nom=import_config["exporter_energy_limit"] / 1e3,
            # in one hour at most 0.1% of total annual energy
        )
        n.add(
            "Link",
            pd.Index(df.importer.unique() + " import infrastructure " + tech),
            bus0=import_buses,
            bus1=domestic_buses,
            carrier="import infrastructure " + tech,
            capital_cost=terminal_capital_cost.get(tech, 0),
            p_nom_extendable=True,
            p_nom_max=p_nom_max,
            p_nom_min=p_nom_min,
            p_min_pu=p_min_pu,
            bus2=bus2,
            efficiency2=efficiency2,
        )

    # need special handling for copperplated imports
    copperplated_options = {
        "shipping-ftfuel",
        "shipping-meoh",
        "shipping-steel",
        "shipping-hbi",
        "shipping-lnh3",
    }

    for tech in set(import_carriers).intersection(copperplated_options):
        marginal_costs = (
            import_costs.query("esc == @tech")
            .groupby("exporter")
            .marginal_cost.min()
            .mul(import_carriers[tech])
        )

        bus2 = "co2 atmosphere" if tech in co2_intensity else ""
        efficiency2 = (
            -costs.at[co2_intensity[tech][0], co2_intensity[tech][1]]
            if tech in co2_intensity
            else np.nan
        )

        # using energy content of iron as proxy: 2.1 MWh/t
        unit_to_mwh = 2.1 if tech in ["shipping-steel", "shipping-hbi"] else 1.0

        suffix = bus_suffix[tech]
        n.add("Carrier", "import " + tech)
        n.add(
            "Link",
            marginal_costs.index + " import " + tech,
            bus0=marginal_costs.index + " export",
            bus1="EU" + suffix,
            carrier="import " + tech,
            p_nom=import_config["exporter_energy_limit"] / 1e3,
            # in one hour at most 0.1% of total annual energy
            marginal_cost=marginal_costs.values / unit_to_mwh,
            efficiency=1 / unit_to_mwh,
            p_min_pu=import_config["min_part_load_shipping_imports"],
            bus2=bus2,
            efficiency2=efficiency2,
        )

        n.add(
            "Link",
            marginal_costs.index + " DE import " + tech,
            bus0=marginal_costs.index + " export",
            bus1="DE" + suffix,
            carrier="import " + tech,
            p_nom=import_config["exporter_energy_limit"] / 1e3,
            # in one hour at most 0.1% of total annual energy
            marginal_cost=marginal_costs.values / unit_to_mwh,
            efficiency=1 / unit_to_mwh,
            p_min_pu=import_config["min_part_load_shipping_imports"],
            bus2=bus2,
            efficiency2=efficiency2,
        )


def relocate_ammonia(n):

    # get nodes
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    EU_nodes = pop_layout[pop_layout.index.str[:2] != "DE"].index
    DE_nodes = pop_layout[pop_layout.index.str[:2] == "DE"].index
    nhours = n.snapshot_weightings.generators.sum()

    # adjust the ammonia loads
    logger.info("Allowing relocation of ammonia industry.")
    # add German NH3 bus
    n.add("Bus", "DE NH3", x=x, y=y, carrier="NH3", unit="MWh_th")
    # TWh to MWh
    industrial_demand = pd.read_csv(snakemake.input.industrial_demand, index_col=0) * 1e6
    p_set = industrial_demand.loc[:, "ammonia"].rename(
                    index=lambda x: x + " NH3").groupby(level=0).sum() / nhours
    p_set_DE = p_set[DE_nodes + " NH3"].sum()
    p_set_EU = p_set[EU_nodes + " NH3"].sum()
    n.add(
            "Load",
            "DE NH3",
            bus="DE NH3",
            carrier="NH3",
            p_set=p_set_DE,
        )
    n.loads.loc["EU NH3", "p_set"] = p_set_EU

    # change bus1 of German Haber-Bosch links to DE NH3
    ammonia_links = n.links[(n.links.carrier == "Haber-Bosch") & (n.links.index.str[:2] == "DE")].index
    n.links.loc[ammonia_links, "bus1"] = "DE NH3"

    # add transport link between DE and EU
    n.add(
        "Link",
        ["DE NH3 -> EU NH3", "EU NH3 -> DE NH3"],
        bus0=["DE NH3", "EU NH3"],
        bus1=["EU NH3", "EU NH3"],
        carrier="NH3",
        p_nom=1e6,
        p_min_pu=0,
        marginal_cost=0.01,
    )
    # copy the ammonia store for the current planning_horizon
    # add stores
    planning_horizon = snakemake.wildcards.planning_horizons
    EU_store = n.stores.loc[f"EU NH3 ammonia store-{planning_horizon}"].copy()
    n.add(
        "Store",
        "DE ammonia store-" + planning_horizon,
        bus="DE methanol",
        carrier="methanol",
        e_nom_extendable=EU_store.e_nom_extendable,
        e_cyclic=EU_store.e_cyclic,
        capital_cost=EU_store.capital_cost,
        overnight_cost=EU_store.overnight_cost,
        lifetime=costs.at["General liquid hydrocarbon storage (product)", "lifetime"],
    )



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "modify_final_network",
            simpl="",
            clusters=68,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2045",
            run="no_import-no_relocation",
        )

    configure_logging(snakemake)
    logger.info("Unravelling remaining import vectors.")

    n = pypsa.Network(snakemake.input.network)
    import_options = snakemake.params.import_options
    country_centroids = pd.read_csv(snakemake.input.country_centroids, index_col="ISO")
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        nyears=1,
    )
    nhours = n.snapshot_weightings.generators.sum()
    sector_options = snakemake.params.sector_options
    relocation_option = sector_options["relocation"]

    if relocation_option in ["ammonia", "all"]:
        relocate_ammonia(n)

    if import_options["enable"] and int(snakemake.wildcards.planning_horizons) >= 2030:
        # all import vectors or only h2 + elec
        if import_options["carriers"] == "all":
            carriers = carriers_all
        elif import_options["carriers"] == "eleh2":
            carriers = carriers_eleh2
        else:
            logger.error("Invalid import carriers option. Must be 'all' or 'eleh2'.")
        # build dictionary with cost factors
        carriers = {k: import_options["cost_factor"] for k in carriers}
        add_import_options(
            n,
            capacity_boost=import_options["capacity_boost"],
            endogenous_hvdc=import_options["endogenous_hvdc_import"]["enable"],
            import_carriers=carriers,
        )

    n.export_to_netcdf(snakemake.output.network)
