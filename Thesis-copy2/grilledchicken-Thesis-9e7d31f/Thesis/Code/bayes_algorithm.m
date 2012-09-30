function [ prob_dist, first_spike, last_spike ] = bayes_algorithm( time, spikes, model_params, time_window, compression_factor, iter_vars, velocity_K)
%function bayes_algorithm(time,gridmax_x, gridmax_y, neurons, spikes, firingrates, spatial_occ, window)
% Function not meant to be called independently. Contains the core
% reconstruction alogrithm. Takes all required data such as firing
% rates and spiking data, along with other algorithm specific
% information such as time window and grid size, and uses them to
% calculate a probability distribution of position.
%
% Output - prob_dist
%
% A matrix of size MxN, containing the probability distribution of expected position.
% MxN is the grid size as specified during training.


gridmax_x=model_params{1}(2);
gridmax_y=model_params{1}(3);
neurons=model_params{1}(1);
spatial_occ=model_params{3};
firingrates=model_params{4};
vel1=model_params{7};
timestep=model_params{1}(4);
count=iter_vars{1};
per_out=iter_vars{2};
first_spike=iter_vars{3};
last_spike=iter_vars{4};
prob_out=iter_vars{5};

ENABLE_CORRECTION=1;


%------------------------------2 step Bayesian Reconstruction implementation----------------------------%

%End points of specified time window
p1=(time- time_window/2);
p2=(time+ time_window/2);

%Preallocate memory  for probability distribution            
% prob_dist=zeros(gridmax_x,gridmax_y);
prob_dist=spatial_occ;



number_of_spikes=0;
for tt=1:neurons
    while(spikes{tt}(first_spike(tt)+1)<p1 && first_spike(tt)<numel(spikes{tt})-1)
        first_spike(tt)=first_spike(tt)+1;
    end
    while(spikes{tt}(last_spike(tt))<p2 && last_spike(tt)<numel(spikes{tt}-1))
        last_spike(tt)=last_spike(tt)+1;
    end
    number_of_spikes(tt)=last_spike(tt)-first_spike(tt)-1;
    if(number_of_spikes<0)
        number_of_spikes=0;
    end

end


for y=1:gridmax_y
    for x=1:gridmax_x

        %CONTINUE if animal never visits this grid box; improves execution time 4X.
        if(spatial_occ(x,y)==0)
            continue;
        end

        %-----Bayes' Theorem implementation (PREDICTION STEP)-----%
        temp=1;    
        temp2=0;
        % fprintf('...............\n');
        for tt=1:neurons
            fr=firingrates{tt}(x,y);
            temp=temp*power(fr,number_of_spikes(tt));
            temp2=temp2+fr;

        end

        temp2=temp2*-time_window;
        temp2=exp(temp2);
        prob_dist(x,y)=spatial_occ(x,y)*temp*temp2;
        %---------------------------------------------------------%

    %----------CORRECTION STEP---------%

    if(ENABLE_CORRECTION==1)
        velocity_constant=velocity_K/vel1(x,y);       
        if(count~=1)
            estx_prev=per_out(count-1,2);
            esty_prev=per_out(count-1,3);
            correction_prob=sqrt(power(estx_prev-x,2)+power(esty_prev-y,2));
            correction_prob=-power(correction_prob,2);
            correction_prob=correction_prob/velocity_constant;
            correction_prob=exp(correction_prob);
            prob_dist(x,y)=prob_dist(x,y)*correction_prob;
        end
    end
    %----------------------------------%

    end 
end


%---Normalize distribution to sum up to 1
total_sum=sum(sum(prob_dist));
if(total_sum~=0)
    normalization_constant=1/total_sum;
    if(normalization_constant==Inf)
        normalization_constant=1;
    end
    prob_dist=prob_dist.*normalization_constant;
end


end

