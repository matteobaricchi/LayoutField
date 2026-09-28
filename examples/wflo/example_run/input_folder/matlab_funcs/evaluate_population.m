function [lcoe_list,ioe_list,cost_per_turbine_list,emission_per_turbine_list,aep_net_factor_list] = evaluate_population(x_fil_list,y_fil_list,z_fil_list,aep_input_list,aep_no_wake_list,power_ilk_list,wd_step_list,TurCon_list,u_list,v_list,cab_data_list,parallel_execution,n_cpu)
% EVALUATE_POPULATION Evaluate a population of layouts using DETECT in parallel
%
% Inputs:
%   x_mat : [n_pop x n_vars] design variables
%
% Outputs:

%% 

% initialize output
n_pop = size(x_fil_list,2);
lcoe_list = cell(1,n_pop);
ioe_list = cell(1,n_pop);
cost_per_turbine_list = cell(1,n_pop);
emission_per_turbine_list = cell(1,n_pop);
aep_net_factor_list = cell(1,n_pop);


if parallel_execution

    parfor (n = 1:n_pop,n_cpu)
    
        x = x_fil_list{n};
        y = y_fil_list{n};
        z = z_fil_list{n};
        aep_input = aep_input_list{n};
        aep_no_wake = aep_no_wake_list{n};
        power_ilk = power_ilk_list{n};
        wd_step = wd_step_list{n};
        TurCon = num2cell(TurCon_list{n});
        u = num2cell(u_list{n});
        v = num2cell(v_list{n});
        cab_data = cab_data_list{n};
    
        aepCalc = 'precalculated';
        cableCalc = 'precalculated';
    
        [LCOE,IOE,DiscountedAnnualCostsPerTurbine,AnnualEmissionsPerTurbine,AEPnetFactor] = RunDetectWithoutPython(x,y,z,aepCalc,aep_input,aep_no_wake,power_ilk,cableCalc,TurCon,u,v,cab_data,wd_step);
    
        lcoe_list{n} = LCOE;
        ioe_list{n} = IOE;
        cost_per_turbine_list{n} = DiscountedAnnualCostsPerTurbine;
        emission_per_turbine_list{n} = AnnualEmissionsPerTurbine;
        aep_net_factor_list{n} = AEPnetFactor;

    end
else
    for n = 1:n_pop
    
        x = x_fil_list{n};
        y = y_fil_list{n};
        z = z_fil_list{n};
        aep_input = aep_input_list{n};
        aep_no_wake = aep_no_wake_list{n};
        power_ilk = power_ilk_list{n};
        wd_step = wd_step_list{n};
        TurCon = num2cell(TurCon_list{n});
        u = num2cell(u_list{n});
        v = num2cell(v_list{n});
        cab_data = cab_data_list{n};
    
        aepCalc = 'precalculated';
        cableCalc = 'precalculated';
    
        [LCOE,IOE,DiscountedAnnualCostsPerTurbine,AnnualEmissionsPerTurbine,AEPnetFactor] = RunDetectWithoutPython(x,y,z,aepCalc,aep_input,aep_no_wake,power_ilk,cableCalc,TurCon,u,v,cab_data,wd_step);
    
        lcoe_list{n} = LCOE;
        ioe_list{n} = IOE;
        cost_per_turbine_list{n} = DiscountedAnnualCostsPerTurbine;
        emission_per_turbine_list{n} = AnnualEmissionsPerTurbine;
        aep_net_factor_list{n} = AEPnetFactor;
        
    end



end
