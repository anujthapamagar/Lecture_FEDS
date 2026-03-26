
###################
### NEW VERSION
###################
# This version only calcualtes the maxmial distance of the crosswind footprint
# This is solving a memory leak in the original version linked with the 2D field.
# 
def FFP(zm=None, z0=None, umean=None, h=None, ol=None, sigmav=None, ustar=None, 
        wind_dir=None, nx=1000,  **kwargs):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015: 
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: n.kljun@swansea.ac.uk

    ******************************************************************
    ** CAUTION CHANGED TO ONLY RETURN FOOTPRINT MAX AND 1D FUNCTION **
    ******************************************************************
    2023 by Steffen Noe


    FFP Input
    zm     = Measurement height above displacement height (i.e. z-d) [m]
    z0     = Roughness length [m]; enter None if not known 
    umean  = Mean wind speed at zm [m/s]; enter None if not known 
             Either z0 or umean is required. If both are given,
             z0 is selected to calculate the footprint
    h      = Boundary layer height [m]
    ol     = Obukhov length [m]
    sigmav = standard deviation of lateral velocity fluctuations [ms-1]
	ustar  = friction velocity [ms-1]

    optional inputs:
    wind_dir = wind direction in degrees (of 360) for rotation of the footprint    
    
    nx       = Integer scalar defining the number of grid elements of the scaled footprint.
               Large nx results in higher spatial resolution and higher computing time.
               Default is 1000, nx must be >=600.

 
    FFP output
    x_ci_max = x location of footprint peak (distance from measurement) [m]
    x_ci	 = x array of crosswind integrated footprint [m]
    f_ci	 = array with footprint function values of crosswind integrated footprint [m-1] 
    flag_err = 0 if no error, 1 in case of error

    created: 15 April 2015 natascha kljun
    translated to python, December 2015 Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.4
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015,2016,2017,2018,2019,2020 Natascha Kljun

    CHANGED to avoid memroy hole by Steffen Noe 30/01/2023 Steffe Noe
    """

    import numpy as np
    import sys
    #import numbers

    #===========================================================================
    ## Input check
    flag_err = 0
        
    ## Check existence of required input pars
    if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1)

    # Check passed values
    if zm <= 0.: raise_ffp_exception(2)
    if z0 is not None and umean is None and z0 <= 0.: raise_ffp_exception(3)
    if h <= 10.: raise_ffp_exception(4)
    if zm > h: raise_ffp_exception(5)        
    if z0 is not None and umean is None and zm <= 12.5*z0:
        raise_ffp_exception(12)
    if float(zm)/ol <= -15.5: raise_ffp_exception(7)
    if sigmav <= 0: raise_ffp_exception(8)
    if ustar <= 0.1: raise_ffp_exception(9)
    if wind_dir is not None:
        if wind_dir > 360. or wind_dir < 0.: raise_ffp_exception(10)
    if nx < 600: raise_ffp_exception(11)

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    if None not in [z0, umean]: raise_ffp_exception(13)

    #===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17 
    bc = 1.66
    cc = 20.0

    xstar_end = 30
    oln = 5000 #limit to L for neutral scaling
    k = 0.4 #von Karman

    #===========================================================================
    # Scaled X* for crosswind integrated footprint
    xstar_ci_param = np.linspace(d, xstar_end, nx+2)
    xstar_ci_param = xstar_ci_param[1:]

    # Crosswind integrated scaled F* 
    fstar_ci_param = a * (xstar_ci_param-d)**b * np.exp(-c/ (xstar_ci_param-d))
    ind_notnan     = ~np.isnan(fstar_ci_param)
    fstar_ci_param = fstar_ci_param[ind_notnan]
    xstar_ci_param = xstar_ci_param[ind_notnan]
    del ind_notnan # not anymore used
    
    # Scaled sig_y*
    sigystar_param = ac * np.sqrt(bc * xstar_ci_param**2 / (1 + cc * xstar_ci_param))

    #===========================================================================
    # Real scale x and f_ci
    if z0 is not None:
        # Use z0
        if ol <= 0 or ol >= oln:
            xx  = (1 - 19.0 * zm/ol)**0.25
            psi_f = np.log((1 + xx**2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi/2
        elif ol > 0 and ol < oln:
            psi_f = -5.3 * zm / ol

        x = xstar_ci_param * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
        if np.log(zm / z0) - psi_f > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f)
        else:
            x_ci_max, x_ci, f_ci, x_2d, y_2d, f_2d = None
            flag_err = 1
    else:
        # Use umean if z0 not available
        x = xstar_ci_param * zm / (1. - zm / h) * (umean / ustar * k)
        if umean / ustar > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * (1. - zm / h) / (umean / ustar * k)
        else:
            x_ci_max, x_ci, f_ci, x_2d, y_2d, f_2d = None
            flag_err = 1
                        
    # Delete not anymore needed objects
    del xstar_ci_param
    
    
    #Maximum location of influence (peak location)
    xstarmax = -c / b + d
    if z0 is not None:
        x_ci_max = xstarmax * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
    else:
        x_ci_max = xstarmax * zm / (1. - (zm / h)) * (umean / ustar * k)

    del xstarmax # not anymore needed
    
    #Real scale sig_y
    if abs(ol) > oln:
        ol = -1E6
    if ol <= 0:   #convective
        scale_const = 1E-5 * abs(zm / ol)**(-1) + 0.80
    elif ol > 0:  #stable
        scale_const = 1E-5 * abs(zm / ol)**(-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0
    sigy = sigystar_param / scale_const * zm * sigmav / ustar
    sigy[sigy < 0] = np.nan
    del sigystar_param # not anymore needed
    
    #Real scale f(x,y)
    dx = x_ci[2] - x_ci[1]
    y_pos = np.arange(0, (len(x_ci) / 2.) * dx * 1.5, dx)
    #f_pos = np.full((len(f_ci), len(y_pos)), np.nan)
    f_pos = np.empty((len(f_ci), len(y_pos)))
    # f_pos[:] = np.nan
    for ix in range(len(f_ci)):
        f_pos[ix,:] = f_ci[ix] * 1 / (np.sqrt(2 * np.pi) * sigy[ix]) * np.exp(-y_pos**2 / ( 2 * sigy[ix]**2))

    #Complete footprint for negative y (symmetrical)
    y_neg = - np.fliplr(y_pos[None, :])[0]
    f_neg = np.fliplr(f_pos)
    y = np.concatenate((y_neg[0:-1], y_pos))
    f = np.concatenate((f_neg[:, :-1].T, f_pos.T)).T
    # delete not anymore needed fields
    del y_neg 
    del f_neg
    
    
    
    #===========================================================================
    # Fill output structure
    
    return {'x_ci_max': x_ci_max, 'x_ci': x_ci, 'f_ci': f_ci, 'flag_err':flag_err}

#===============================================================================



#===============================================================================
exTypes = {'message': 'Message',
           'alert': 'Alert',
           'error': 'Error',
           'fatal': 'Fatal error'}

exceptions = [
    {'code': 1,
     'type': exTypes['fatal'],
     'msg': 'At least one required parameter is missing. Please enter all '
            'required inputs. Check documentation for details.'},
    {'code': 2,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be larger than zero.'},
    {'code': 3,
     'type': exTypes['error'],
     'msg': 'z0 (roughness length) must be larger than zero.'},
    {'code': 4,
     'type': exTypes['error'],
     'msg': 'h (BPL height) must be larger than 10 m.'},
    {'code': 5,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be smaller than h (PBL height).'},
    {'code': 6,
     'type': exTypes['alert'],
     'msg': 'zm (measurement height) should be above roughness sub-layer (12.5*z0).'},
    {'code': 7,
     'type': exTypes['error'],
     'msg': 'zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.'},
    {'code': 8,
     'type': exTypes['error'],
     'msg': 'sigmav (standard deviation of crosswind) must be larger than zero.'},
    {'code': 9,
     'type': exTypes['error'],
     'msg': 'ustar (friction velocity) must be >=0.1.'},
    {'code': 10,
     'type': exTypes['error'],
     'msg': 'wind_dir (wind direction) must be >=0 and <=360.'},
    {'code': 11,
     'type': exTypes['fatal'],
     'msg': 'Passed data arrays (ustar, zm, h, ol) don\'t all have the same length.'},
    {'code': 12,
     'type': exTypes['fatal'],
     'msg': 'No valid zm (measurement height above displacement height) passed.'},
    {'code': 13,
     'type': exTypes['alert'],
     'msg': 'Using z0, ignoring umean if passed.'},
    {'code': 14,
     'type': exTypes['alert'],
     'msg': 'No valid z0 passed, using umean.'},
    {'code': 15,
     'type': exTypes['fatal'],
     'msg': 'No valid z0 or umean array passed.'},
    {'code': 16,
     'type': exTypes['error'],
     'msg': 'At least one required input is invalid. Skipping current footprint.'},
    {'code': 17,
     'type': exTypes['alert'],
     'msg': 'Only one value of zm passed. Using it for all footprints.'},
    # {'code': 18,
    #  'type': exTypes['fatal'],
    #  'msg': 'if provided, rs must be in the form of a number or a list of numbers.'},
    # {'code': 19,
    #  'type': exTypes['alert'],
    #  'msg': 'rs value(s) larger than 90% were found and eliminated.'},
    {'code': 20,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be above roughness sub-layer (12.5*z0).'},
    ]

def raise_ffp_exception(code):
    '''Raise exception or prints message according to specified code'''
	
    ex = [it for it in exceptions if it['code'] == code][0]
    string = ex['type'] + '(' + str(ex['code']).zfill(4) + '):\n '+ ex['msg'] 

    print('')
    if ex['type'] == exTypes['fatal']:
        string = string + '\n FFP_fixed_domain execution aborted.'
        raise Exception(string)
    else:
        print(string)

