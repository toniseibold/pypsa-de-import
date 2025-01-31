# -*- coding: utf-8 -*-
import logging
import pandas as pd
from xarray import DataArray

from scripts.prepare_sector_network import determine_emission_sectors

logger = logging.getLogger(__name__)


def force_boiler_profiles_existing_per_boiler(n):
    """
    This scales each boiler dispatch to be proportional to the load profile.
    """

    logger.info(
        "Forcing each existing boiler dispatch to be proportional to the load profile"
    )

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    # will be per unit
    n.model.add_variables(coords=[decentral_boilers], name="Link-fixed_profile_scaling")

    lhs = (1, n.model["Link-p"].loc[:, decentral_boilers]), (
        -boiler_profiles,
        n.model["Link-fixed_profile_scaling"],
    )

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile_scaling")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.links["fixed_profile_scaling_opt"] = 0.0


def add_co2limit_country(n, limit_countries, snakemake, debug=False):
    """
    Add a set of emissions limit constraints for specified countries.

    The countries and emissions limits are specified in the config file entry 'co2_budget_national'.

    Parameters
    ----------
    n : pypsa.Network
    limit_countries : dict
    snakemake: snakemake object
    """
    logger.info(f"Adding CO2 budget limit for each country as per unit of 1990 levels")

    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    sectors = determine_emission_sectors(n.config["sector"])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in limit_countries:
        limit = co2_total_totals[ct] * limit_countries[ct]
        logger.info(
            f"Limiting emissions in country {ct} to {limit_countries[ct]:.1%} of "
            f"1990 levels, i.e. {limit:,.2f} tCO2/a",
        )

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:
            links = n.links.index[
                (n.links.index.str[:2] == ct)
                & (n.links[f"bus{port}"] == "co2 atmosphere")
                & (
                    n.links.carrier != "kerosene for aviation"
                )  # first exclude aviation to multiply it with a domestic factor later
            ]

            logger.info(
                f"For {ct} adding following link carriers to port {port} CO2 constraint: {n.links.loc[links,'carrier'].unique()}"
            )

            if port == "0":
                efficiency = -1.0
            elif port == "1":
                efficiency = n.links.loc[links, f"efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            lhs.append(
                (
                    n.model["Link-p"].loc[:, links]
                    * efficiency
                    * n.snapshot_weightings.generators
                ).sum()
            )

        # Aviation demand
        energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])
        domestic_aviation = energy_totals.loc[
            ("DE", snakemake.params.energy_year), "total domestic aviation"
        ]
        international_aviation = energy_totals.loc[
            ("DE", snakemake.params.energy_year), "total international aviation"
        ]
        domestic_factor = domestic_aviation / (
            domestic_aviation + international_aviation
        )
        aviation_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "kerosene for aviation")
        ]
        lhs.append
        (
            n.model["Link-p"].loc[:, aviation_links.index]
            * aviation_links.efficiency2
            * n.snapshot_weightings.generators
        ).sum() * domestic_factor
        logger.info(
            f"Adding domestic aviation emissions for {ct} with a factor of {domestic_factor}"
        )
        # Toni TODO: add non European import as well!
        # Adding Efuel imports and exports to constraint
        incoming_oil = n.links.index[n.links.index == "EU renewable oil -> DE oil"]
        non_eu_oil = n.links.index[(n.links.carrier=="import shipping-ftfuel") & (n.links.bus1=="DE renewable oil")]
        incoming_oil = incoming_oil.append(non_eu_oil)
        outgoing_oil = n.links.index[n.links.index == "DE renewable oil -> EU oil"]

        if not debug:
            lhs.append(
                (
                    -1
                    * n.model["Link-p"].loc[:, incoming_oil]
                    * 0.2571
                    * n.snapshot_weightings.generators
                ).sum()
            )
            lhs.append(
                (
                    n.model["Link-p"].loc[:, outgoing_oil]
                    * 0.2571
                    * n.snapshot_weightings.generators
                ).sum()
            )

        incoming_methanol = n.links.index[n.links.index == "EU methanol -> DE methanol"]
        non_eu_methanol = n.links.index[(n.links.carrier=="import shipping-meoh") & (n.links.bus1=="DE methanol")]
        incoming_methanol = incoming_methanol.append(non_eu_methanol)
        outgoing_methanol = n.links.index[n.links.index == "DE methanol -> EU methanol"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_methanol]
                / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_methanol]
                / snakemake.config["sector"]["MWh_MeOH_per_tCO2"]
                * n.snapshot_weightings.generators
            ).sum()
        )

        # Methane
        incoming_CH4 = n.links.index[n.links.index == "EU renewable gas -> DE gas"]
        non_eu_CH4 = n.links.index[(n.links.carrier=="import infrastructure shipping-lch4") & (n.links.bus1=="DE renewable gas")]
        incoming_CH4 = incoming_CH4.append(non_eu_CH4)
        outgoing_CH4 = n.links.index[n.links.index == "DE renewable gas -> EU gas"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def import_limit_eu(n, sns, limit_eu_de, investment_year):
    """
    Limiting European imports to Germany to net 0 TWh for each carrier.
    """

    rhs = 0

    logger.info("Limiting European imports to Germany to net 0 TWh for each carrier.")
    ct = "DE"
    # collect indices of more complex carriers
    h2_in = n.links.index[
        (n.links.carrier.str.contains("H2 pipeline"))
        & (n.links.bus0.str[:2] != ct)
        & (n.links.bus1.str[:2] == ct)
    ]
    h2_out = n.links.index[
        (n.links.carrier.str.contains("H2 pipeline"))
        & (n.links.bus0.str[:2] == ct)
        & (n.links.bus1.str[:2] != ct)
    ]
    elec_links_in = n.links.index[
        ((n.links.carrier == "DC") | (n.links.carrier == "AC"))
        & (n.links.bus0.str[:2] != ct)
        & (n.links.bus1.str[:2] == ct)
    ]
    elec_links_out = n.links.index[
        ((n.links.carrier == "DC") | (n.links.carrier == "AC"))
        & (n.links.bus0.str[:2] == ct)
        & (n.links.bus1.str[:2] != ct)
    ]

    elec_lines_in = n.lines.index[
        (n.lines.carrier == "AC")
        & (n.lines.bus0.str[:2] != ct)
        & (n.lines.bus1.str[:2] == ct)
    ]
    elec_lines_out = n.lines.index[
        (n.lines.carrier == "AC")
        & (n.lines.bus0.str[:2] == ct)
        & (n.lines.bus1.str[:2] != ct)
    ]

    lhs_ele_s = (
        n.model["Line-s"].loc[sns, elec_lines_in].sum()
        - n.model["Line-s"].loc[sns, elec_lines_out].sum()
    )
    lhs_ele_p = (
        n.model["Link-p"].loc[sns, elec_links_in].sum()
        - n.model["Link-p"].loc[sns, elec_links_out].sum()
    )

    lhs_h2 = (
        n.model["Link-p"].loc[sns, h2_in].sum()
        - n.model["Link-p"].loc[sns, h2_out].sum()
    )

    lhs_ft = (
        n.model["Link-p"].loc[sns, "EU renewable oil -> DE oil"].sum()
        - n.model["Link-p"].loc[sns, "DE renewable oil -> EU oil"].sum()
    )

    lhs_gas = (
        n.model["Link-p"].loc[sns, "EU renewable gas -> DE gas"].sum()
        - n.model["Link-p"].loc[sns, "DE renewable gas -> EU gas"].sum()
    )

    # electricity
    n.model.add_constraints(lhs_ele_s, "==", rhs, name="import_limit_lines")
    n.model.add_constraints(lhs_ele_p, "==", rhs, name="import_limit_ele")
    # hydrogen
    n.model.add_constraints(lhs_h2, "==", rhs, name="import_limit_h2")
    # h2 derivatives
    n.model.add_constraints(lhs_ft, "==", rhs, name="import_limit_ft")
    n.model.add_constraints(lhs_gas, "==", rhs, name="import_limit_gas")


def import_limit_non_eu(n, sns, limit_non_eu_de, investment_year):

    logger.info("Adding a limit of {limit_non_eu_de} TWh of non-European imports.")

    # get all non-European import links
    non_eu_links = n.links[
        (n.links.bus1.str[:2] == "DE") & (n.links.carrier.str.contains("import"))
    ].index

    if non_eu_links.empty:
        logger.warning(
            "No non-European import links found but limit_non_eu_de is set. Please check config[solving][constraints][limit_non_eu_de] and config[import][enable]."
        )
        return

    weightings = n.snapshot_weightings.loc[sns, "generators"]

    p_links = n.model["Link-p"].loc[sns, non_eu_links]

    lhs = (p_links * weightings).sum()

    rhs = limit_non_eu_de * 1e6

    n.model.add_constraints(lhs, "==", rhs, name="energy_import_limit")

    # restrict hydrogen export
    h2_links = n.links.index[
        (n.links.bus0.str[:2] == "DE")
        & (n.links.bus1.str[:2] != "DE")
        & (n.links.carrier.str.contains("H2 pipeline"))
    ]
    lhs = (
        n.model["Link-p"].loc[sns, h2_links] * n.snapshot_weightings.generators
    ).sum()

    n.model.add_constraints(lhs, "<=", 0, name="h2_export_limit_DE")

    # might be necessary to add constraint for electricity export as well


def ramp_up_limit_non_EU(n, n_snapshots, limits_volume_max, investment_year):

    if investment_year not in limits_volume_max["h2_derivate_import"]["DE"].keys():
        return
    limit = limits_volume_max["h2_derivate_import"]["DE"][investment_year] * 1e6

    logger.info(f"limiting non European H2 derivate imports to DE to {limit/1e6} TWh/a")

    non_eu_links = n.links[
        (n.links.bus1.str[:2] == "DE")
        & (n.links.carrier.str.contains("import"))
        & ~(n.links.carrier.str.contains("h2"))
    ].index

    incoming_p = (
        n.model["Link-p"].loc[:, non_eu_links] * n.snapshot_weightings.generators
    ).sum()

    lhs = incoming_p

    cname = "non_European_H2_derivate_import_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if investment_year not in limits_volume_max["h2_import"]["DE"].keys():
        return
    limit = limits_volume_max["h2_import"]["DE"][investment_year] * 1e6

    logger.info(f"limiting non European H2 imports to DE to {limit/1e6} TWh/a")

    non_eu_links = n.links[
        (n.links.bus1.str[:2] == "DE")
        & (n.links.carrier.str.contains("import"))
        & (n.links.carrier.str.contains("h2"))
    ].index

    incoming_p = (
        n.model["Link-p"].loc[:, non_eu_links] * n.snapshot_weightings.generators
    ).sum()

    lhs = incoming_p

    cname = "non_European_H2_import_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")



def additional_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    constraints = snakemake.params.solving["constraints"]

    if isinstance(constraints["co2_budget_national"], dict):
        limit_countries = constraints["co2_budget_national"][investment_year]
        add_co2limit_country(
            n,
            limit_countries,
            snakemake,
            debug=snakemake.config["run"]["debug_co2_limit"],
        )
    else:
        logger.warning("No national CO2 budget specified!")
    # enforce boilers to follow heat demand
    force_boiler_profiles_existing_per_boiler(n)

    limit_eu_de = constraints["limit_eu_de"]
    limit_non_eu_de = constraints["limit_non_eu_de"]

    if limit_eu_de:
        logger.info("Adding import limit for European imports to Germany.")
        import_limit_eu(n, snapshots, limit_eu_de, investment_year)
    if "import shipping-lh2" in n.links.carrier.unique():
        logger.info("Ramp up import limit for non European imports to Germany.")
        ramp_up_limit_non_EU(
            n, snapshots, constraints["limits_volume_max"], investment_year
        )
    if limit_non_eu_de:
        logger.info("Adding import limit for non European imports to Germany.")
        import_limit_non_eu(n, snapshots, limit_non_eu_de, investment_year)
