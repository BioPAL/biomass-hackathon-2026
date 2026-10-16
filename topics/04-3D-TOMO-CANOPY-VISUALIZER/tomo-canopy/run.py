from tomo_canopy.visualisation.gedi_helper import get_gedi_points
from tomo_canopy.visualisation.tls_helper import get_tls_point_data, get_tls_vol_data
from tomo_canopy.visualisation.tomographic_plotter import DataGetter, TomographicPlotter


def main():
    tls_getter = DataGetter(
        get_vol_data_fn=get_tls_vol_data, get_points_fn=get_tls_point_data
    )
    gedi_getter = DataGetter(get_vol_data_fn=None, get_points_fn=get_gedi_points)
    TomographicPlotter(
        {
            "TLS": tls_getter,
            "GEDI": gedi_getter,
        }
    ).run()


if __name__ == "__main__":
    main()
