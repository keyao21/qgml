"""
MOdule for outputing plots comparing trajectories
"""
import argparse 
import os 
import numpy as np 
import matplotlib.pyplot as plt
import util 
import double_gyre as dg
import interp 
from config import * 


def generate_single_trajectory( velocity_func_filenames, initial_conditions, 
                                elapsed_time, dt, showfig=True, savefig=False):
    """
    Create trajectory plots for single intiial condition
    Loads stream function file and calculate trajectories
    using RK4
    
    velocity_func_filenames: list of string filename(s)
    initial_conditions: tuple of initial condition coordinates
    """
    num_samples = 1
    vfuncs_list = [] 
    for velocity_func_filename in velocity_func_filenames:
        velocity_func_fullpath = os.path.join( INTERP_VELOCITY_PATH_DIR, velocity_func_filename)
        vfuncs_list.append( util.load_velocity_field(velocity_func_fullpath) )
    # Generate estimated trajectories
    # state : numpy array shaped as ...
    # ( num velocity_funcs, num_samples, time steps, num coordinates)
    state = np.zeros(( len(velocity_func_filenames), num_samples, int(elapsed_time/dt)+1, 2 ))
    for vfunc_idx, vfuncs in enumerate(vfuncs_list):
        state[vfunc_idx,:,0,0] = initial_conditions[0]
        state[vfunc_idx,:,0,1] = initial_conditions[1]
        for i,t in enumerate(np.arange(0, elapsed_time, dt)):
            # import pdb;pdb.set_trace()
            # print( state[vfunc_idx,:,(i-max(5,i)):i,:] )
            state[vfunc_idx,:,i+1,:] = interp.rk4(vfuncs, state[vfunc_idx,:,i,:].copy(), t, dt)
        print('Trajectories finished.')

    plt.figure()
    plt.subplot(2,1,1)
    for i,_ in enumerate(vfuncs_list):
        print("plotting x")
        plt.plot(state[i,0,:,0].transpose())
        stable_t = check_stable(state[i,0,:,:])
        if stable_t: plt.axvline(x=stable_t)

    plt.subplot(2,1,2)
    for i,_ in enumerate(vfuncs_list):
        print("plotting y")
        plt.plot(state[i,0,:,1].transpose())
        stable_t = check_stable(state[i,0,:,:])
        if stable_t: plt.axvline(x=stable_t)


    
    if savefig:
        # save plots 
        trajectory_dir_fullpath = os.path.join(RESULT_PATH_DIR,"trajectories")
        util.set_up_dir(trajectory_dir_fullpath)

        filename = velocity_func_filenames[0].strip('.est').strip('.actual').strip('.uvinterp') \
                 + '_' + str(initial_conditions[0]) + '_' + str(initial_conditions[1]) + '.png'
        
        trajectory_file_fullpath = os.path.join(trajectory_dir_fullpath, filename)
        plt.savefig(trajectory_file_fullpath)


    if showfig: plt.show()

    plt.close('all')
    return state


def run_game( velocity_func_filenames, elapsed_time, dt, dim, showfig=True): 
    """
    dim is the dimension which determines which gyre (usually 0 or 1, 0 for double gyre)

    This method is an implementation of another way to measure "accuracy" of one velcotiy field 
    to another.  The inspiration is from real life rescue missions where people typically use 
    statistics to predict where a something (or someone!) that fell overboard in the middle of the ocean 
    would be after some time. 

    Nowadays, it might seem smarter to make predictions using models with more physical intuition. Granted, 
    we aren't using that much intuition in training our model (compared to typical fluid models that mathematical.
    physically intensive), our model takes into account a lot of input from the system and tries to learn and 
    reverse engineer the system's behavior and mechanics (by infering the stream function and likewise the velcoity fields). 
    So it's reasonable to say it's a slight step up from basic statistical inference.  
    
    Our "game" here will be the following steps: 

    1. select a bunch of points throughout the field, initialize as noninertial particles @ t=0
    2. at multiple t>0, calculate each particles' new position based on and compare: 
        a. actual velocity field 
        b. estimated velocity field 
    3. aggregate results at the end, measuring how closely particles move in the estimated
       velocity field compared to the acutal velocity field. 


    In step 2, we need a good way to compare. For now, we'll settle for just whether they're in the same gyre.  
    """


    # TODO: factor this logic out of run_game and generate_single_trajectory
    num_samples = 20000
    vfuncs_list = [] 
    for velocity_func_filename in velocity_func_filenames:
        velocity_func_fullpath = os.path.join( INTERP_VELOCITY_PATH_DIR, velocity_func_filename)
        vfuncs_list.append( util.load_velocity_field(velocity_func_fullpath) )
        # TODO: MULTIPLE VFUNCS PER VELOCITY_FUNC_FULLPATH
        #!!!!

    # Generate estimated trajectories
    # state : numpy array shaped as ...
    # ( num velocity_funcs, num_samples, time steps, num coordinates)
    state = np.zeros(( len(velocity_func_filenames), num_samples, int(elapsed_time/dt)+1, 2 ))
    for vfunc_idx, vfuncs in enumerate(vfuncs_list):
        # set initial conditions: range of values for x, set value for y
        # state[vfunc_idx,:,0,0] = np.linspace(0.99995, 1.0005, num_samples, np.float64)
        # state[vfunc_idx,:,0,1] = 0.5*np.ones(num_samples, np.float64)

        np.random.seed(0)
        state[vfunc_idx,:,0,dim] = np.random.uniform(0.2,1.8,num_samples)
        state[vfunc_idx,:,0,abs(dim-1)] = np.random.uniform(0.15,0.85,num_samples)

        for i,t in enumerate(np.arange(0, elapsed_time, dt)):
            # import pdb;pdb.set_trace()
            # print( state[vfunc_idx,:,(i-max(5,i)):i,:] )
            state[vfunc_idx,:,i+1,:] = interp.rk4(vfuncs, state[vfunc_idx,:,i,:].copy(), t, dt)
        print('Trajectories finished.')

 

    # check_stable_vfunc = np.vectorize(check_stable)


    # plt.figure()
    total_scores, total_observations = np.array([]), np.array([])
    for sample_i in range(num_samples): 
        est_stable_t_i = check_stable(state[0,sample_i,:,:])
        act_stable_t_i = check_stable(state[1,sample_i,:,:])
        min_stable_ti = min(est_stable_t_i, act_stable_t_i)

        # note that this is only the dimension which has gyre
        unstable_est_state_i = state[0,sample_i,:min_stable_ti,dim].copy()
        unstable_act_state_i = state[1,sample_i,:min_stable_ti,dim].copy()

        list_scores = evaluate([unstable_est_state_i, unstable_act_state_i])
        list_observations = np.ones(list_scores.shape)
        total_scores = util.add_numpy_arrays(total_scores, list_scores)
        total_observations = util.add_numpy_arrays(total_observations, list_observations)
    


        # import pdb; pdb.set_trace()
        # plt.figure()
        # plt.plot(unstable_est_state_i)
        # plt.plot(unstable_act_state_i)
        # plt.show()

    # import pdb;pdb.set_trace()




    if showfig: 
        fig = plt.figure()
        plt.plot(total_scores/total_observations)    
        plt.show()

    return total_scores, total_observations


def OLD_evaluate(list_of_state_vectors, length=2.0):
    """
    Helper function to evaluate/compare vectors 
    list_of_state_vectors is a list of *two* numpy vectors (state_vector)
    length is the total range of the axis (half exactly will mark the difference between gyres)
        - usually its just 2.0 

    state_vector is a vector: < timestep, dimension >. 
        - Note that this state_vector IS NOT of the same data structure as "state" in generate_single_trajectory(). 
          Rather, it's a cross section, taking...
          **only ONE dimension (x or y)**
          **only ONE of the trajectories (est or actual)**
          
    returns list of 0s and 1s, representing fails and passes
    """
    normed_state_vectors = [state_vector - (length/2.) for state_vector in list_of_state_vectors]
    multiply_state_vectors = normed_state_vectors[0] *normed_state_vectors[1]
    positive_ct = len(np.where(multiply_state_vectors > 0.0)[0])
    negative_ct = len(np.where(multiply_state_vectors < 0.0)[0])

    func_01 = lambda x: 0. if (x < 0.) else 1.
    vec_func_01 = np.vectorize(func_01)
    return vec_func_01(multiply_state_vectors)


def evaluate(list_of_state_vectors, length=2.0):
    """
    Helper function to evaluate/compare vectors 
    list_of_state_vectors is a list of *two* numpy vectors (state_vector)
    length is the total range of the axis (half exactly will mark the difference between gyres)
        - usually its just 2.0 

    state_vector is a vector: < timestep, dimension >. 
        - Note that this state_vector IS NOT of the same data structure as "state" in generate_single_trajectory(). 
          Rather, it's a cross section, taking...
          **only ONE dimension (x or y)**
          **only ONE of the trajectories (est or actual)**
          
    returns list of 0s and 1s, representing fails and passes
    """
    normed_state_vectors = [state_vector - (length/2.) for state_vector in list_of_state_vectors]

    

    
    multiply_state_vectors = normed_state_vectors[0] *normed_state_vectors[1]
    positive_ct = len(np.where(multiply_state_vectors > 0.0)[0])
    negative_ct = len(np.where(multiply_state_vectors < 0.0)[0])

    func_01 = lambda x: 0. if (x < 0.) else 1.
    vec_func_01 = np.vectorize(func_01)
    return vec_func_01(multiply_state_vectors)




def check_stable(state_vector, epsilon=1e-4): 
    """
    Helper function to determine sublength of vector "state_vector" that is "unstable" and not at 
    either a stable fixed point or boundary (sticky), as determined SOLELY by its x and y velocity. 

    state_vector is a vector: < timestep, dimension >. 
        - Note that this state_vector IS NOT of the same data structure as "state" in generate_single_trajectory(). 
          Rather, it's a cross section, taking...
          **only ONE dimension (x or y)**
          **only ONE of the trajectories (est or actual)**
          
    returns int (length of vector that is unstable)
    """
    # first and second order position deltas (similar to derivatives)
    first_order_diff_state_vector = np.diff( state_vector, n=1, axis=0 )
    second_order_diff_state_vector = np.diff( state_vector, n=1, axis=0 )
    

    # lagged first/second order position deltas 
    lag1_first_order_diff_state_vector = util.shift_numpy_array(arr=first_order_diff_state_vector, num=1)
    lag1_second_order_diff_state_vector = util.shift_numpy_array(arr=second_order_diff_state_vector, num=1)
    lag2_first_order_diff_state_vector = util.shift_numpy_array(arr=first_order_diff_state_vector, num=2)
    lag2_second_order_diff_state_vector = util.shift_numpy_array(arr=second_order_diff_state_vector, num=2)
    lag3_first_order_diff_state_vector = util.shift_numpy_array(arr=first_order_diff_state_vector, num=3)
    lag3_second_order_diff_state_vector = util.shift_numpy_array(arr=second_order_diff_state_vector, num=3)

    # sum of first two orders and lagged
    diff_state_vector =    abs(first_order_diff_state_vector)      + abs(second_order_diff_state_vector) \
                         + abs(lag1_first_order_diff_state_vector) + abs(lag1_second_order_diff_state_vector) \
                         + abs(lag2_first_order_diff_state_vector) + abs(lag2_second_order_diff_state_vector) \
                         + abs(lag3_first_order_diff_state_vector) + abs(lag3_second_order_diff_state_vector)


    _x = np.where(abs(diff_state_vector[:,0])<=epsilon)[0]
    _y = np.where(abs(diff_state_vector[:,1])<=epsilon)[0]
    try: 
        first_occur_x = _x[0]
        first_occur_y = _y[0]
    except IndexError: 
        print("no stable point found.")
        return state_vector.shape[0]
    # print(_x, _y)
    return max(first_occur_x, first_occur_y)


def compare_single_trajectories(  ): 
    pass
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', '--trajectories', '-t', type=int)
    parser.add_argument('--test', type=int)
    args = parser.parse_args()

    if args.test == 1: 
        velocity_func_filenames = ['dgsf_0p1_128_64_0p1_0p25000_2.0.uvinterp.est', 'dgsf_0p1_128_64_0p1_0p25000_2.0.uvinterp.actual']
        # velocity_func_filenames = ['QGds02di02dm02p3_1000_1.4.uvinterp.est', 'QGds02di02dm02p3_1000_1.4.uvinterp.actual']
        initial_conditions = [1.25, 0.25]
        elapsed_time = 250
        dt = 0.05
        generate_single_trajectory(velocity_func_filenames=velocity_func_filenames, initial_conditions=initial_conditions,
                                   elapsed_time=elapsed_time, dt=dt, savefig=True )

    elif args.test == 2: 
        # for loop covering all the ML param permutations
        # for spectral_radius in [1.33, 1.66, 2.0]:
        for spectral_radius in [2.33, 2.66, 3.0, 3.33]:
        # for spectral_radius in [3.66, 4.0]: 
            for resSize in [5000]: 

                # get experiment names to retrieve velocity function files (both est and actual)

                dg_params_prefix = 'dgsf_0.1_160_80_0.1_0.2'
                experiment_prefix = f"{dg_params_prefix}_{resSize}_{spectral_radius:.1f}"
                velocity_func_filenames = [ f"{experiment_prefix}.uvinterp.{actual_flag}" for actual_flag in ['est', 'actual']  ]
                print(velocity_func_filenames)

                initial_conditions = [1.0, 0.5]
                elapsed_time = 250
                dt = 0.1
                generate_single_trajectory(velocity_func_filenames=velocity_func_filenames, initial_conditions=initial_conditions,
                                           elapsed_time=elapsed_time, dt=dt, showfig=True, savefig=True )


    elif args.test == 3:
        dg_params_prefix = 'dgsf_0.1_160_80_0.1_0.2'
        resSize, spectral_radius = 5000, 3.7
        experiment_prefix = f"{dg_params_prefix}_{resSize}_{spectral_radius:.1f}"
        velocity_func_filenames = [ f"{experiment_prefix}.uvinterp.{actual_flag}" for actual_flag in ['est', 'actual']  ]
        initial_conditions = [1.0, 0.5]
        elapsed_time = 250
        dt = 0.1
        state = generate_single_trajectory(velocity_func_filenames=velocity_func_filenames, initial_conditions=initial_conditions,
                                            elapsed_time=elapsed_time, dt=dt, showfig=True, savefig=True )
        

        # est_state_0 = state[0,0,:,:]
        # stable_t = check_stable(est_state_0)
    elif args.test == 4: 
        dg_params_prefix = 'dgsf_0.1_160_80_0.1_0.2'
        resSize, spectral_radius = 5000, 3.3
        experiment_prefix = f"{dg_params_prefix}_{resSize}_{spectral_radius:.1f}"
        velocity_func_filenames = [ f"{experiment_prefix}.uvinterp.{actual_flag}" for actual_flag in ['est', 'actual']  ]
        # initial_conditions = [1.0, 0.5]
        elapsed_time = 250
        dt = 0.1
        dim = 0
        run_game( velocity_func_filenames, elapsed_time, dt, dim)

    elif args.test == 5:
        ## DOUBLE GYRE TESTING
        all_game_res = dict()
        for spectral_radius in [1.33, 1.66, 2.0, 2.33, 2.66, 3.0, 3.33, 3.66, 4.0]:
            for resSize in [5000]: 
                dg_params_prefix = 'dgsf_0.1_160_80_0.1_0.2'
                experiment_prefix = f"{dg_params_prefix}_{resSize}_{spectral_radius:.1f}"
                velocity_func_filenames = [ f"{experiment_prefix}.uvinterp.{actual_flag}" for actual_flag in ['est', 'actual']  ]
                elapsed_time = 250
                dt = 0.1
                dim = 0
                scores,observations = run_game( velocity_func_filenames, elapsed_time, dt, dim, showfig=False)
                all_game_res[f"resSize{resSize}_spectralRadius{spectral_radius}"] = [scores,observations]

        fig = plt.figure()
        for key, (scores,observations) in all_game_res.items(): plt.plot(scores/observations, label=key)
        plt.legend(loc="best")
        plt.ylabel("% hit")
        plt.xlabel("time steps")
        import pdb;pdb.set_trace()

    elif args.test == 6:
        ## QUASIGEO TESTING
        all_game_res = dict()



        # ml params 
        resSize = 1000
        spectral_radius = 6.0
        noise = 1e-2
        input_scaling = 0.3
        ridgeReg = 0.01
        density = 0.1
        leaking_rate = 0.5


        # qg params 
        xct = 80
        yct = 160
        ds = 0.03
        di = 0.0
        dm = 0.0
        pertamp = 0.3
        dt = 0.01
        elapsed_time = 10
        input_scaling = 0.1

        for ridgeReg in [0.1, 0.01, 1.0]: 
            qg_params_prefix = f"QGds{ds:.2f}di{di:.2f}dm{dm:.2f}p{pertamp:.1f}"
            experiment_prefix = f"{qg_params_prefix}rs{resSize}sr{spectral_radius:.1f}dens{density:.1f}lr{leaking_rate:.1f}insc{input_scaling:.1f}reg{ridgeReg:.1f}"
            velocity_func_filenames = [ f"{experiment_prefix}.uvinterp.{actual_flag}" for actual_flag in ['est', 'actual']  ]
            dim = 1
            scores,observations = run_game( velocity_func_filenames, elapsed_time, dt, dim, showfig=False)
            all_game_res[f"ridgeReg{ridgeReg}"] = [scores,observations]

        fig = plt.figure()
        for key, (scores,observations) in all_game_res.items(): plt.plot(scores/observations, label=key)
        plt.legend(loc="best")
        plt.ylabel("% hit")
        plt.xlabel("time steps")
        import pdb;pdb.set_trace()


        # SMOOTH: 
        # plt.plot( util.moving_average_smooth(scores/observations, 30) )
        # plt.show()




