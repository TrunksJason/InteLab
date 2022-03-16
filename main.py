import pandas as pd
import numpy as np
import pvlib
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')

cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']

cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

location = Location(latitude=44.9612, longitude=37.2849)

system = PVSystem(surface_tilt=20, surface_azimuth=200,
                  module_parameters=sandia_module,
                  inverter_parameters=cec_inverter,
                  temperature_model_parameters=temperature_model_parameters)

mc = ModelChain(system, location)
weather = pd.read_csv(r"C:\Users\LAPTOP GAME VIP\Desktop\INTERLAB\WeatherData.csv")
#DataFrame([[1050, 1000, 100, 30, 5]],
#                       columns=['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed'],
#                       index=[pd.Timestamp('20170401 1200', tz='US/Arizona')])
print(weather)
mc.run_model(weather)
mc.results.aoi
mc.results.cell_temperature
mc.results.dc
mc.results.ac

location = Location(44.9612,37.2849)

poorly_specified_system = PVSystem()

print(location)

print(poorly_specified_system)
ModelChain(poorly_specified_system, location)
sapm_system = PVSystem(
    module_parameters=sandia_module,
    inverter_parameters=cec_inverter,
    temperature_model_parameters=temperature_model_parameters)

mc = ModelChain(sapm_system, location)

print(mc)
mc.run_model(weather);

mc.results.ac
pvwatts_system = PVSystem(
    module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
    inverter_parameters={'pdc0': 240},
    temperature_model_parameters=temperature_model_parameters)


mc = ModelChain(pvwatts_system, location,
                aoi_model='physical', spectral_model='no_loss')

print(mc)
sapm_system = PVSystem(
    module_parameters=sandia_module,
    inverter_parameters=cec_inverter,
    temperature_model_parameters=temperature_model_parameters)


mc = ModelChain(sapm_system, location, aoi_model='physical', spectral_model='no_loss')

print(mc)

mc.run_model(weather);

mc.results.ac
mc = mc.with_sapm(sapm_system, location)

print(mc)

mc.run_model(weather)

mc.results.dc

mc.run_model()

mc.pvwatts_dc()
pvwatts_system = PVSystem(
    module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
    inverter_parameters={'pdc0': 240},
    temperature_model_parameters=temperature_model_parameters)

mc = ModelChain(pvwatts_system, location,
                aoi_model='no_loss', spectral_model='no_loss')

mc.results.effective_irradiance = pd.Series(1000, index=[pd.Timestamp('20170401 1200-0700')])

mc.results.cell_temperature = pd.Series(50, index=[pd.Timestamp('20170401 1200-0700')])


mc.pvwatts_dc();

mc.results.dc

mc.sapm()

sapm_system = PVSystem(
    module_parameters=sandia_module,
    inverter_parameters=cec_inverter,
    temperature_model_parameters=temperature_model_parameters)


mc = ModelChain(sapm_system, location)



mc.results.effective_irradiance = pd.Series(1000, index=[pd.Timestamp('20170401 1200-0700')])

mc.results.cell_temperature = pd.Series(50, index=[pd.Timestamp('20170401 1200-0700')])


mc.sapm();

mc.results.dc
pvwatts_system = PVSystem(
    module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
    inverter_parameters={'pdc0': 240},
    temperature_model_parameters=temperature_model_parameters)

mc = ModelChain(pvwatts_system, location,
                aoi_model='no_loss', spectral_model='no_loss')


mc.dc_model.__func__

mc.infer_dc_model()
mc.infer_ac_model()
pvlib.modelchain._snl_params()
pvlib.modelchain._adr_params()
pvlib.modelchain._pvwatts_params()
#======================================
from pvlib.pvsystem import Array

location = Location(latitude=44.9612, longitude=37.2849)

inverter_parameters = {'pdc0': 10000, 'eta_inv_nom': 0.96}

module_parameters = {'pdc0': 250, 'gamma_pdc': -0.004}

array_one = Array(mount=FixedMount(surface_tilt=20, surface_azimuth=200),
                  module_parameters=module_parameters,
                  temperature_model_parameters=temperature_model_parameters,
                  modules_per_string=10, strings=2)

array_two = Array(mount=FixedMount(surface_tilt=20, surface_azimuth=160),
                  module_parameters=module_parameters,
                  temperature_model_parameters=temperature_model_parameters,
                  modules_per_string=10, strings=2)

system_two_arrays = PVSystem(arrays=[array_one, array_two],
                             inverter_parameters={'pdc0': 8000})

mc = ModelChain(system_two_arrays, location, aoi_model='no_loss',
                spectral_model='no_loss')

mc.run_model(weather)


mc.results.dc

mc.results.dc[0]
#===========================================================
def pvusa(poa_global, wind_speed, temp_air, a, b, c, d):
    """
    Calculates system power according to the PVUSA equation
    P = I * (a + b*I + c*W + d*T)
    where
    P is the output power,
    I is the plane of array irradiance,
    W is the wind speed, and
    T is the temperature
    a, b, c, d are empirically derived parameters.
    """
    return poa_global * (a + b*poa_global + c*wind_speed + d*temp_air)


def pvusa_mc_wrapper(mc):
    """
    Calculate the dc power and assign it to mc.results.dc
    Set up to iterate over arrays and total_irrad. mc.system.arrays is
    always a tuple. However, when there is a single array
    mc.results.total_irrad will be a Series (if multiple arrays,
    total_irrad will be a tuple). In this case we put total_irrad
    in a list so that we can iterate. If we didn't put total_irrad
    in a list, iteration will access each value of the Series, one
    at a time.
    The iteration returns a tuple. If there is a single array, the
    tuple is of length 1. As a convenience, pvlib unwraps tuples of length 1
    that are assigned to ModelChain.results attributes.
    Returning mc is optional, but enables method chaining.
    """
    if mc.system.num_arrays == 1:
        total_irrads = [mc.results.total_irrad]
    else:
        total_irrads = mc.results.total_irrad
    mc.results.dc = tuple(
        pvusa(total_irrad['poa_global'], mc.results.weather['wind_speed'],
              mc.results.weather['temp_air'], array.module_parameters['a'],
              array.module_parameters['b'], array.module_parameters['c'],
              array.module_parameters['d'])
        for total_irrad, array
        in zip(total_irrads, mc.system.arrays))
    return mc


def pvusa_ac_mc(mc):
    mc.results.ac = mc.results.dc
    return mc


def no_loss_temperature(mc):
    mc.results.cell_temperature = mc.results.weather['temp_air']
    return mc
module_parameters = {'a': 0.2, 'b': 0.00001, 'c': 0.001, 'd': -0.00005}

pvusa_system = PVSystem(module_parameters=module_parameters)

mc = ModelChain(pvusa_system, location,
                dc_model=pvusa_mc_wrapper, ac_model=pvusa_ac_mc,
                temperature_model=no_loss_temperature,
                aoi_model='no_loss', spectral_model='no_loss')
mc.dc_model.func
mc = mc.run_model(weather)
mc.results.dc

