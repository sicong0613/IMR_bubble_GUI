function [struct_vars] = fun_fitting_vars_modification(struct_vars, struct_vars_2modified)

    for i = 1:numel(struct_vars_2modified)
        for j = 1:numel(struct_vars)
            if strcmp(struct_vars(j).name, struct_vars_2modified(i).name)
                struct_vars(j).value = struct_vars_2modified(i).value;
                struct_vars(j).lb = struct_vars_2modified(i).lb;
                struct_vars(j).ub = struct_vars_2modified(i).ub;
                struct_vars(j).scale = struct_vars_2modified(i).scale;
                struct_vars(j).group = struct_vars_2modified(i).group;
            end
        end
    end
end