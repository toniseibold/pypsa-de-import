import pypsa
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import logging
from types import SimpleNamespace

import sys
import os
paths = ["workflow/submodules/pypsa-eur/scripts", "../submodules/pypsa-eur/scripts"]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))
from prepare_sector_network import prepare_costs

logger = logging.getLogger(__name__)

carriers_eleh2 = ["pipeline-h2",
                "shipping-lh2",
                "hvdc-to-elec"]

carriers_all = ["pipeline-h2",
                "shipping-lh2",
                "shipping-lch4",
                "shipping-lnh3",
                "shipping-ftfuel",
                "shipping-meoh",
                "hvdc-to-elec",
                "shipping-hbi",
                "shipping-steel"]

x = 10.5
y = 51.2


def add_endogenous_hvdc_import_options(n, cost_factor=1.0):
    logger.info("Add import options: endogenous hvdc-to-elec")
    cf = snakemake.config["sector"]["import"].get("endogenous_hvdc_import", {})
    if not cf["enable"]:
        return

    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")

    p_max_pu = xr.open_dataset(snakemake.input.import_p_max_pu).p_max_pu.sel(
        importer="EUE"
    )

    p_nom_max = (
        xr.open_dataset(snakemake.input.import_p_max_pu)
        .p_nom_max.sel(importer="EUE")
        .to_pandas()
    )

    def _coordinates(ct):
        iso2 = ct.split("-")[0]
        if iso2 in country_centroids.index:
            return country_centroids.loc[iso2, ["longitude", "latitude"]].values
        else:
            query = cc.convert(iso2, to="name")
            loc = geocode(dict(country=query), language="en")
            return [loc.longitude, loc.latitude]

    exporters = pd.DataFrame(
        {ct: _coordinates(ct) for ct in cf["exporters"]}, index=["x", "y"]
    ).T
    geometry = gpd.points_from_xy(exporters.x, exporters.y)
    exporters = gpd.GeoDataFrame(exporters, geometry=geometry, crs=4326)

    import_links = {}
    a = regions.representative_point().to_crs(DISTANCE_CRS)

    # Prohibit routes through Russia or Belarus
    forbidden_hvdc_importers = ["FI", "LV", "LT", "EE"]
    a = a.loc[~a.index.str[:2].isin(forbidden_hvdc_importers)]

    for ct in exporters.index:
        b = exporters.to_crs(DISTANCE_CRS).loc[ct].geometry
        d = a.distance(b)
        import_links[ct] = (
            d.where(d < d.quantile(cf["distance_threshold"])).div(1e3).dropna()
        )  # km
    import_links = pd.concat(import_links)
    import_links.loc[
        import_links.index.get_level_values(0).str.contains("KZ|CN")
    ] *= 1.2  # proxy for detour through Caucasus

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

    import_links = pd.concat([import_links, pd.Series(xlinks)], axis=0)
    import_links = import_links.drop_duplicates(keep="first")
    duplicated = import_links.index.duplicated(keep="first")
    import_links = import_links.loc[~duplicated]

    hvdc_cost = (
        import_links.values * cf["length_factor"] * costs.at["HVDC submarine", "fixed"]
        + costs.at["HVDC inverter pair", "fixed"]
    )

    buses_i = exporters.index

    n.madd("Bus", buses_i, **exporters.drop("geometry", axis=1))

    efficiency = cf["efficiency_static"] * cf["efficiency_per_1000km"] ** (
        import_links.values / 1e3
    )

    n.madd(
        "Link",
        ["import hvdc-to-elec " + " ".join(idx).strip() for idx in import_links.index],
        bus0=import_links.index.get_level_values(0),
        bus1=import_links.index.get_level_values(1),
        carrier="import hvdc-to-elec",
        p_min_pu=0,
        p_nom_extendable=True,
        length=import_links.values,
        capital_cost=hvdc_cost * cost_factor,
        efficiency=efficiency,
        p_nom_max=cf["p_nom_max"],
    )

    for tech in ["solar-utility", "onwind"]:
        p_max_pu_tech = p_max_pu.sel(technology=tech).to_pandas().dropna().T

        exporters_tech_i = exporters.index.intersection(p_max_pu_tech.columns)

        grid_connection = costs.at["electricity grid connection", "fixed"]

        n.madd(
            "Generator",
            exporters_tech_i,
            suffix=" " + tech,
            bus=exporters_tech_i,
            carrier=f"external {tech}",
            p_nom_extendable=True,
            capital_cost=(costs.at[tech, "fixed"] + grid_connection) * cost_factor,
            lifetime=costs.at[tech, "lifetime"],
            p_max_pu=p_max_pu_tech.reindex(columns=exporters_tech_i),
            p_nom_max=p_nom_max[tech].reindex(index=exporters_tech_i)
            * cf["share_of_p_nom_max_available"],
        )

    # hydrogen storage

    h2_buses_i = n.madd(
        "Bus", buses_i, suffix=" H2", carrier="external H2", location=buses_i
    )

    n.madd(
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

    n.madd(
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

    n.madd(
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

    b_buses_i = n.madd(
        "Bus", buses_i, suffix=" battery", carrier="external battery", location=buses_i
    )

    n.madd(
        "Store",
        b_buses_i,
        bus=b_buses_i,
        carrier="external battery",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=costs.at["battery storage", "fixed"] * cost_factor,
        lifetime=costs.at["battery storage", "lifetime"],
    )

    n.madd(
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

    n.madd(
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

    for bus0_bus1 in cf.get("extra_connections", []):
        bus0, bus1 = bus0_bus1.split("-")

        a = exporters.to_crs(DISTANCE_CRS).at[bus0, "geometry"]
        b = exporters.to_crs(DISTANCE_CRS).at[bus1, "geometry"]
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

def add_import_options(n, capacity_boost, import_options, endogenous_hvdc=False):

    if not snakemake.config["sector"]["gas_network"]:
        logger.warning(
            "Gas network is not resolved but recommended for import options."
        )

    logger.info("Add import options: " + " ".join(import_options.keys()))
    fn = snakemake.input.gas_input_nodes_simplified
    import_nodes = pd.read_csv(fn, index_col=0)
    import_nodes["hvdc-to-elec"] = 15000

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
        "shipping-lch4": " gas",
        "shipping-lnh3": " NH3",
        "shipping-ftfuel": " oil",
        "shipping-meoh": " methanol",
        "shipping-steel": " steel",
        "shipping-hbi": " hbi",
    }

    co2_intensity = {
        "shipping-lch4": ("gas", "CO2 intensity"),
        "shipping-ftfuel": ("oil", "CO2 intensity"),
        "shipping-meoh": ("methanolisation", "carbondioxide-input"),
    }

    # import_costs = pd.read_csv(
    #     snakemake.input.import_costs, index_col=0)
    
    
    import_costs = pd.read_csv(
        snakemake.input.import_costs, delimiter=";", keep_default_na=False
    )

    # temporary bugfix for Namibia
    import_costs["exporter"] = import_costs.exporter.replace("", "NA")

    cols = ["esc", "exporter", "importer", "value"]
    fields = ["Cost per MWh delivered", "Cost per t delivered"]
    import_costs = import_costs.query("subcategory in @fields")[cols]
    import_costs.rename(columns={"value": "marginal_cost"}, inplace=True)

    for k, v in translate.items():
        import_nodes[k] = import_nodes[v]
        ports[k] = ports.get(v)

    # TODO: Check how and if this works
    if endogenous_hvdc and "hvdc-to-elec" in import_options:
        add_endogenous_hvdc_import_options(n, import_options.pop("hvdc-to-elec"))

    regionalised_options = {
        "hvdc-to-elec",
        "pipeline-h2",
        "shipping-lh2",
        "shipping-lch4",
    }

    for tech in set(import_options).intersection(regionalised_options):
        import_costs_tech = (
            import_costs.query("esc == @tech").groupby("importer").marginal_cost.min()
        ) * import_options[tech]

        sel = ~import_nodes[tech].isna()
        if tech == "pipeline-h2":
            entrypoints_internal = ["DE", "BE", "FR", "GB"]
            entrypoints_via_RU_BY = ["EE", "LT", "LV", "FI"]  # maybe PL Yamal
            forbidden_pipelines = entrypoints_internal + entrypoints_via_RU_BY
            sel &= ~import_nodes.index.str[:2].isin(forbidden_pipelines)
        import_nodes_tech = import_nodes.loc[sel, [tech]]

        import_nodes_tech = import_nodes_tech.rename(columns={tech: "p_nom"})

        marginal_costs = ports[tech].dropna().map(import_costs_tech)
        import_nodes_tech["marginal_cost"] = marginal_costs

        import_nodes_tech.dropna(inplace=True)

        suffix = bus_suffix[tech]

        if tech in co2_intensity.keys():
            buses = import_nodes_tech.index + f"{suffix} import {tech}"
            capital_cost = 7018.0 if tech == "shipping-lch4" else 0.0  # €/MW/a

            n.madd("Bus", buses + " bus", carrier=f"import {tech}")
            # TODO:
            n.madd(
                "Store",
                buses + " store",
                bus=buses + " bus",
                e_nom_extendable=True,
                e_nom_min=-np.inf,
                e_nom_max=0,
                e_min_pu=1,
                e_max_pu=0,
            )
            if snakemake.config["sector"]["gas_network"]:
                bus1 = import_nodes_tech.index + suffix
            else:
                bus1 = ["EU gas"]
            n.madd(
                "Link",
                buses,
                bus0=buses + " bus",
                bus1=bus1,
                bus2="co2 atmosphere",
                carrier=f"import {tech}",
                efficiency2=-costs.at[co2_intensity[tech][0], co2_intensity[tech][1]],
                marginal_cost=import_nodes_tech.marginal_cost.values,
                p_nom_extendable=True,
                capital_cost=capital_cost,
            )

        else:
            location = import_nodes_tech.index
            buses = location if tech == "hvdc-to-elec" else location + suffix

            capital_cost = (
                1.2 * 7018.0 if tech == "shipping-lh2" else 0.0
            )  # €/MW/a, +20% compared to LNG

            n.madd(
                "Generator",
                import_nodes_tech.index + f"{suffix} import {tech}",
                bus=buses,
                carrier=f"import {tech}",
                marginal_cost=import_nodes_tech.marginal_cost.values,
                p_nom_extendable=True,
                capital_cost=capital_cost,
            )

    # need special handling for copperplated imports
    copperplated_carbonaceous_options = {
        "shipping-ftfuel",
        "shipping-meoh",
    }
    # add for EU and DE
    for tech in set(import_options).intersection(copperplated_carbonaceous_options):
        marginal_costs = (
            import_costs.query("esc == @tech").marginal_cost.min()
            * import_options[tech]
        )

        suffix = bus_suffix[tech]

        n.add("Bus", f"EU import {tech} bus", carrier=f"import {tech}")
        n.add("Bus", f"DE import {tech} bus", carrier=f"import {tech}")

        n.add(
            "Store",
            f"EU import {tech} store",
            bus=f"EU import {tech} bus",
            e_nom_extendable=True,
            e_nom_min=-np.inf,
            e_nom_max=0,
            e_min_pu=1,
            e_max_pu=0,
        )

        n.add(
            "Store",
            f"DE import {tech} store",
            bus=f"DE import {tech} bus",
            e_nom_extendable=True,
            e_nom_min=-np.inf,
            e_nom_max=0,
            e_min_pu=1,
            e_max_pu=0,
        )

        n.add(
            "Link",
            f"EU import {tech}",
            bus0=f"EU import {tech} bus",
            bus1="EU" + suffix,
            bus2="co2 atmosphere",
            carrier=f"import {tech}",
            efficiency2=-costs.at[co2_intensity[tech][0], co2_intensity[tech][1]],
            marginal_cost=marginal_costs,
            p_nom=1e7,
        )
        n.add(
            "Link",
            f"DE import {tech}",
            bus0=f"DE import {tech} bus",
            bus1="DE" + suffix,
            bus2="co2 atmosphere",
            carrier=f"import {tech}",
            efficiency2=-costs.at[co2_intensity[tech][0], co2_intensity[tech][1]],
            marginal_cost=marginal_costs,
            p_nom=1e7,
        )

    copperplated_carbonfree_options = {
        "shipping-steel",
        "shipping-hbi",
        "shipping-lnh3",
    }

    # add for EU and DE
    for tech in set(import_options).intersection(copperplated_carbonfree_options):
        suffix = bus_suffix[tech]

        marginal_costs = (
            import_costs.query("esc == @tech").marginal_cost.min()
            * import_options[tech]
        )
        n.add(
            "Generator",
            f"EU import {tech}",
            bus="EU" + suffix,
            carrier=f"import {tech}",
            marginal_cost=marginal_costs,
            p_nom=1e7,
        )
        n.add(
            "Generator",
            f"DE import {tech}",
            bus="DE" + suffix,
            carrier=f"import {tech}",
            marginal_cost=marginal_costs,
            p_nom=1e7,
        )

def unravel_ammonia(n, costs, sector_options):
    
    # unravel ammonia
    if not sector_options["ammonia"]:
        logger.error("Ammonia sector must be activated. Please set config['sector']['ammonia'] to True.")

    logger.info("Unraveling ammonia")
    n.add("Bus", "DE NH3", carrier="NH3", x=x, y=y)

    # add ammonia store
    n.add(
        "Store",
        "DE ammonia store",
        bus="DE NH3",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="ammonia store",
        capital_cost=costs.at["NH3 (l) storage tank incl. liquefaction", "fixed"],
        overnight_cost=costs.at[
            "NH3 (l) storage tank incl. liquefaction", "investment"
        ],
        lifetime=costs.at["NH3 (l) storage tank incl. liquefaction", "lifetime"],
    )

    # ammonia links
    HB_links = n.links[(n.links.carrier=="Haber-Bosch") & (n.links.index.str.contains("DE"))].index
    n.links.loc[HB_links, "bus1"] = "DE NH3"
    crack_links = n.links[(n.links.carrier=="ammonia cracker") & (n.links.index.str.contains("DE"))].index
    n.links.loc[crack_links, "bus0"] = "DE NH3"

    # transport links
    n.madd(
        "Link",
        ["EU NH3 -> DE NH3", "DE NH3 -> EU NH3"],
        bus0=["EU NH3", "DE NH3"],
        bus1=["DE NH3", "EU NH3"],
        carrier="NH3",
        p_nom=1e4,
        marginal_cost=0.1,
        p_min_pu=0,
    )
    # adjust loads
    industrial_demand = get_industrial_demand() * 1e6 # TWh/a -> MWh/a
    industrial_demand_DE = industrial_demand[industrial_demand.index.get_level_values(0).str.contains("DE")]
    DE_ammonia = industrial_demand_DE.loc[:, "ammonia"].sum() / 8760

    n.add("Load", "DE NH3 load", bus="DE NH3", carrier="NH3", p_set=DE_ammonia)
    n.loads.loc["EU NH3", "p_set"] -= DE_ammonia


def get_industrial_demand():
    # import ratios [MWh/t_Material]
    fn = snakemake.input.industry_sector_ratios
    sector_ratios = pd.read_csv(fn, header=[0, 1], index_col=0)

    # material demand per node and industry [kt/a]
    fn = snakemake.input.industrial_production_per_node
    nodal_production = pd.read_csv(fn, index_col=0) / 1e3 # kt/a -> Mt/a

    nodal_sector_ratios = pd.concat(
        {node: sector_ratios[node[:2]] for node in nodal_production.index}, axis=1
    )

    nodal_production_stacked = nodal_production.stack()
    nodal_production_stacked.index.names = [None, None]

    # final energy consumption per node and industry (TWh/a)
    nodal_df = (
        (nodal_sector_ratios.multiply(nodal_production_stacked))
        .T
    )

    return nodal_df


def endogenise_steel(n, costs, sector_options):

    # industrial demand [TWh/a]
    industrial_demand = get_industrial_demand() * 1e6 # TWh/a -> MWh/a
    # industrial production in [kt/a]
    industrial_production = (pd.read_csv(snakemake.input.industrial_production, index_col=0) * 1e3) # kt/a -> t/a

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    DE_nodes = pop_layout[pop_layout.index.str[:2] == "DE"].index
    EU_nodes = pop_layout[pop_layout.index.str[:2] != "DE"].index
    nodes = pop_layout.index

    endogenous_sectors = []

    if not sector_options["steel"]["endogenous"]:
        logger.error("Endogenous steel demand must be activated. Please set config['sector']['steel']['endogenous'] to True.")

    logger.info("Adding endogenous primary steel demand in tonnes.")

    sector = "DRI + Electric arc"
    endogenous_sectors += [sector]

    no_relocation = not sector_options["steel"]["relocation"]
    no_flexibility = not sector_options["steel"]["flexibility"]

    s = " not" if no_relocation else " "
    logger.info(f"Steel industry relocation{s} activated.")

    s = " not" if no_flexibility else " "
    logger.info(f"Steel industry flexibility{s} activated.")

    # add steel bus for DE and EU
    n.add(
        "Bus",
        "EU steel",
        carrier="steel",
        unit="t",
    )
    n.add(
        "Bus",
        "DE steel",
        x=x,
        y=y,
        carrier="steel",
        unit="t",
    )

    n.add(
        "Bus",
        "EU hbi",
        carrier="hbi",
        unit="t",
    )
    n.add(
        "Bus",
        "DE hbi",
        x=x,
        y=y,
        carrier="hbi",
        unit="t",
    )
    # load EU and DE steel
    n.add(
        "Load",
        "EU steel",
        bus="EU steel",
        carrier="steel",
        p_set=industrial_production.loc[EU_nodes][sector].sum() / 8760,
    )
    n.add(
        "Load",
        "DE steel",
        bus="DE steel",
        carrier="steel",
        p_set=industrial_production.loc[DE_nodes][sector].sum() / 8760,
    )

    if not no_flexibility:
        n.add(
            "Store",
            "EU steel Store",
            bus="EU steel",
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="steel",
        )
        n.add(
            "Store",
            "DE steel Store",
            bus="DE steel",
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="steel",
        )
        n.add(
            "Store",
            "EU hbi Store",
            bus="EU hbi",
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="hbi",
        )
        n.add(
            "Store",
            "DE hbi Store",
            bus="DE hbi",
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="hbi",
        )

    #  Iron ore to hbi
    electricity_input = costs.at["direct iron reduction furnace", "electricity-input"]
    hydrogen_input = costs.at["direct iron reduction furnace", "hydrogen-input"]

    # so that for each region supply matches consumption
    p_nom_EU = (
        industrial_production[industrial_production.index.str[:2] !="DE"][sector] 
        * costs.at["electric arc furnace", "hbi-input"] 
        * electricity_input
        / 8760
    )
    p_nom_DE = (
        industrial_production[industrial_production.index.str[:2] =="DE"][sector]
        * costs.at["electric arc furnace", "hbi-input"]
        * electricity_input
        / 8760
    )
    marginal_cost = (
            costs.at["iron ore DRI-ready", "commodity"]
            * costs.at["direct iron reduction furnace", "ore-input"]
            / electricity_input
        )

    n.madd(
        "Link",
        EU_nodes,
        suffix=" DRI",
        carrier="DRI",
        capital_cost=costs.at["direct iron reduction furnace", "fixed"]
        / electricity_input,
        marginal_cost=marginal_cost,
        p_nom=p_nom_EU if no_relocation else 0,
        p_nom_extendable=False if no_relocation else True,
        bus0=EU_nodes,
        bus1="EU hbi",
        bus2=EU_nodes + " H2",
        efficiency=1 / electricity_input,
        efficiency2=-hydrogen_input / electricity_input,
    )
    n.madd(
        "Link",
        DE_nodes,
        suffix=" DRI",
        carrier="DRI",
        capital_cost=costs.at["direct iron reduction furnace", "fixed"]
        / electricity_input,
        marginal_cost=marginal_cost,
        p_nom=p_nom_DE if no_relocation else 0,
        p_nom_extendable=False if no_relocation else True,
        bus0=DE_nodes,
        bus1="DE hbi",
        bus2=DE_nodes + " H2",
        efficiency=1 / electricity_input,
        efficiency2=-hydrogen_input / electricity_input,
    )

    # HBI to steel via electric arc furnace
    electricity_input = costs.at["electric arc furnace", "electricity-input"]

    p_nom_EU = (
        industrial_production[industrial_production.index.str[:2] !="DE"][sector]
        * electricity_input
        / 8760
    )
    p_nom_DE = (
        industrial_production[industrial_production.index.str[:2] =="DE"][sector]
        * electricity_input
        / 8760
    )

    n.madd(
        "Link",
        EU_nodes,
        suffix=" EAF",
        carrier="EAF",
        capital_cost=costs.at["electric arc furnace", "fixed"] / electricity_input,
        p_nom=p_nom_EU if no_relocation else 0,
        p_nom_extendable=False if no_relocation else True,
        bus0=EU_nodes,
        bus1="EU steel",
        bus2="EU hbi",
        efficiency=1 / electricity_input,
        efficiency2=-costs.at["electric arc furnace", "hbi-input"]
        / electricity_input,
    )
    n.madd(
        "Link",
        DE_nodes,
        suffix=" EAF",
        carrier="EAF",
        capital_cost=costs.at["electric arc furnace", "fixed"] / electricity_input,
        p_nom=p_nom_DE if no_relocation else 0,
        p_nom_extendable=False if no_relocation else True,
        bus0=DE_nodes,
        bus1="DE steel",
        bus2="DE hbi",
        efficiency=1 / electricity_input,
        efficiency2=-costs.at["electric arc furnace", "hbi-input"]
        / electricity_input,
    )

    # allow transport of steel between EU and DE
    n.madd(
        "Link",
        ["EU steel -> DE steel", "DE steel -> EU steel", "EU hbi -> DE hbi", "DE hbi -> EU hbi"],
        bus0=["EU steel", "DE steel", "EU hbi", "DE hbi"],
        bus1=["DE steel", "EU steel", "DE hbi", "EU hbi"],
        carrier=["steel", "steel", "hbi", "hbi"],
        p_nom=1e3,
        marginal_cost=0.1,
        p_min_pu=0,
    )

    adjust_industry_loads(n, nodes, industrial_demand, endogenous_sectors)


def adjust_industry_loads(n, nodes, industrial_demand, endogenous_sectors):

    #TODO: get in a bunch of error messages if configs are not right

    # readjust the loads from industry without steel and potentially hvc
    remaining_sectors = ~industrial_demand.index.get_level_values(1).isin(endogenous_sectors)
    
    remaining_demand = (
        industrial_demand.loc[(nodes, remaining_sectors), :]
        .groupby(level=0)
        .sum()
    )

    # methane
    gas_demand = (
        remaining_demand.loc[:, "methane"]
            .groupby(level=0)
            .sum()
            .rename(index=lambda x: x + " gas for industry")
            / 8760
        )

    n.loads.loc[gas_demand.index, "p_set"] = gas_demand.values

    # hydrogen
    h2_demand = (
        remaining_demand.loc[:, "hydrogen"]
            .groupby(level=0)
            .sum()
            .rename(index=lambda x: x + " H2 for industry")
            / 8760
    )

    n.loads.loc[h2_demand.index, "p_set"] = h2_demand.values

    # heat
    heat_demand = (
        remaining_demand.loc[:, "heat"]
            .groupby(level=0)
            .sum()
            .rename(index=lambda x: x + " low-temperature heat for industry")
            / 8760
    )
    n.loads.loc[heat_demand.index, "p_set"] = heat_demand.values

    # elec
    elec_demand = (
        remaining_demand.loc[:, "elec"]
            .groupby(level=0)
            .sum()
            .rename(index=lambda x: x + " industry electricity")
            / 8760
    )
    n.loads.loc[elec_demand.index, "p_set"] = elec_demand.values

    # process emission
    process_emissions = (
            -remaining_demand.loc[:, "process emission"]
            .groupby(level=0)
            .sum()
            .rename(index=lambda x: x + " process emissions")
            / 8760
        )

    n.loads.loc[process_emissions.index, "p_set"] = process_emissions.values

    if sector_options["co2network"]:
        logger.error("CO2 network not working yet. Please add code to the function adjust_industry_loads().")

    

if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "modify_final_network",
            simpl="",
            clusters=27,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2030",
            run="KN2045_Bal_v4",
        )

    logger.info("Unravelling remaining import vectors.")

    n = pypsa.Network(snakemake.input.network)
    import_options = snakemake.params.import_options
    country_centroids = pd.read_csv(snakemake.input.country_centroids, index_col="ISO")
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        nyears=1,
    )
    sector_options = snakemake.params.sector_options

    endogenise_steel(n, costs, sector_options)

    unravel_ammonia(n, costs, sector_options)

    if import_options["enable"]:
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
            import_options=carriers,
            endogenous_hvdc=import_options["endogenous_hvdc_import"]["enable"],
        )

    n.export_to_netcdf(snakemake.output.network)