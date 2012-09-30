function [ model_params ] = training( position_data, spikes, binsize_grid, intervals )
%
% [model_params] = training(position_data, spikes, binsize_grid, intervals)
%
% Trains the model on given position data and spiking activity of an neural
% ensemble. The grid size for discretization is also specified. Optionally,
% intervals within which training should be carried out can also be specified.
%
% Inputs -
% position_data - Positional data in the form of a Tx3 matrix, where T is the
%                 number of timesteps. The three columns correspond to timestep,
%                 X coordinate, and the Y coordinate at the timestep respectively.
% spikes        - A cell array containing N vectors, where N is the number of
%                 neurons. Each vector contains timestamps at which the neuron
%                 fired.
%
% Optional Inputs-
% binsize_grid  - [M,N] - Is a vector containing two values, M and N, and is used
%                 to discretize the data into an MxN grid. If only [M] is specified,
%                 an MxM grid is used. By default, a 32x32 bin density is used.
% intervals     - This is a Ix2 matrix, where I is the number of intervals.
%                 The model will be trained only on data falling within these 
%                 intervals. At every interval specified, the first column 
%                 represents the start timestamp, and the second columnt 
%                 indicates the end timestamp for the interval. By default,
%                 A single interval that encompasses all given data is used.
%
% Outputs-
% model_params  - Model_params contains parameters built from the given data.
%                 It is in the form of a cell array, that contains the following
%                 general elements : number of neurons, grid size X, grid size Y.
%                 It also contains occupancy matrix and firing rates for the 
%                 individual neurons. The intervals between which it was trained
%                 is also contained in model_params. The model_params cell array
%                 encapsulates all information that would be required for
%                 reconstruction by any algorithm.

%%%-----delete any spikes occuring before the first position timestamp
minpost=min(position_data(:,1));
dels=0;
for x=1:numel(spikes)
    while(spikes{x}(1)<minpost)
        dels=dels+1;
        spikes{x}(1)=[];
    end
end
if(dels>0)
    fprintf('%d spikes that occur before minimum position timestamp IGNORED\n',dels);
end
%%--------------------------------------------------------------------------------




if(nargin<2)
    error('Need atleast position data and spiking information');
elseif(nargin<3)
    binsize_grid=[32]; % 32x32 default;
    intervals=[min(position_data(:,1)),max(position_data(:,1))];
elseif(nargin<4)
    intervals=[min(position_data(:,1)),max(position_data(:,1))];
end

intervals=[min(position_data(:,1)),min(position_data(:,1));intervals];



if(numel(binsize_grid)==1)
    x=binsize_grid(1);
    binsize_grid=[x,x];
end
binsize_grid=binsize_grid+0;    % Correction for zeroth bin error, during calculation and display.
spike_window_tolerance=2;       % A spike is counted within interval if it occurs within these many timesteps. #default=2

max_x=max(position_data(:,2));  % get max X value
max_y=max(position_data(:,3));  % get max Y value
n_grid=binsize_grid(1);       % horizontal divisions, n
m_grid=binsize_grid(2);       % vertical divisions, m
if(n_grid<2 || m_grid<2)
    error('Minimum grid size should be 2x2'); % minimum 4x4
end
m_grid=max_x/m_grid;            % bin width
n_grid=max_y/n_grid;            % bin height

fprintf('Discretizing into %dx%d grid...\n',binsize_grid(1),binsize_grid(2));
%------------------Discretize position data into bins, as per given grid size
for x=1:numel(position_data(:,1))
    position_data(x,2)=round(position_data(x,2)/m_grid);
    position_data(x,3)=round(position_data(x,3)/n_grid);
end
max_x=max(position_data(:,2));
max_y=max(position_data(:,3));


%--------------------Create new data set 'posdata', containing position data contained ONLY within the specified intervals
posdata=[];
for tempx=1:numel(intervals(:,1))
    startpoint=findnearest(intervals(tempx,1),position_data(:,1),1); %added the postive search gain 1
    if(numel(startpoint)==0)
        startpoint=findnearest(intervals(tempx,1),position_data(:,1),-1); %if nothing ahead, look behind
    end
    startpoint=startpoint(1);
    endpoint=findnearest(intervals(tempx,2),position_data(:,1),-1); %added the negative search gain -1
    endpoint=endpoint(1);   
    posdata=[posdata;position_data(startpoint:endpoint,:)];
end


ignore_orig=0;  % Set to 1, to ignore all (0,0) points

tempy=[];
for tempx=2:numel(posdata(:,1))
    tempy=[tempy;posdata(tempx,1)-posdata(tempx-1,1)];
end
del_t=(mean(tempy));


gridmax_x=max_x;
gridmax_y=max_y;

fprintf('Calculating velocity...\n');
%=================VELOCITY AT EVERY GRID CELL=========%
vel1=zeros(gridmax_x,gridmax_y);     % MAXIMUM TIME SPENT CONSECUTIVELY AT ANY LOCATION
vel2=zeros(gridmax_x,gridmax_y);     % TIME SPENT AT EACH ITER
vel3=zeros(gridmax_x,gridmax_y);     % AVERAGE TIME SPENT AT EACH LOCATION
velcount=ones(gridmax_x,gridmax_y);  % initialized to ones, instead of zeroes for better maps.

changed=0;
previous_x=posdata(1,2);
previous_y=posdata(1,3);
for x=1:numel(posdata(:,1))
    current_x=posdata(x,2);
    current_y=posdata(x,3);
    if(current_x==0)
        current_x=1;
    end
    if(current_y==0)
        current_y=1;
    end
    if(current_y==previous_y && current_x==previous_x)
        vel2(current_x,current_y)=vel2(current_x,current_y)+1;
    else
        vel3(current_x,current_y)=vel3(current_x,current_y)+vel2(current_x,current_y);
        velcount(current_x,current_y)=velcount(current_x,current_y)+1;

        if(vel2(current_x,current_y)>vel1(current_x,current_y))
            vel1(current_x,current_y)=vel2(current_x,current_y);
            vel2(current_x,current_y)=0;
        end
    previous_x=current_x;
    previous_y=current_y;
    end
end
vel3=vel3./velcount;  





%=============== SPATIAL OCCUPANCY ===================%
fprintf('Calculating Firing rates...\n');
spatial_occ=zeros(gridmax_x,gridmax_y);
for x=1:numel(posdata(:,1))
    xx=posdata(x,2);
    yy=posdata(x,3);
    if(ignore_orig==1)
        if(xx==1 && yy==1)
            continue;
        end
    end
    xx=floor(xx);
    yy=floor(yy);
    if(xx==0)
        xx=1;
    end
    if(yy==0)
        yy=1;
    end
    spatial_occ(xx,yy)=spatial_occ(xx,yy)+1;
end
total_positions=sum(sum(spatial_occ));
occupancy_matrix=spatial_occ;
spatial_occ=spatial_occ./total_positions;

%================== FIRING RATES ====================%
% waitb=waitbar(0,'Calculating Firing Rates...');

neurons=numel(spikes);
firingrates={};

sz=size(posdata);
sz=sz(1);

for x=1:neurons
    frate=zeros(gridmax_x,gridmax_y);
    for timestamp=1:size(spikes{x})

        % index=findnearest(spikes{x}(timestamp),posdata(:,1));   
        % index=index(1)


        if(timestamp==1)
            index=findnearest(spikes{x}(timestamp),posdata(:,1));   
            index=index(1);
        else
            val=index;
            temp_time=spikes{x}(timestamp);
            while(posdata(val,1)<temp_time)
                val=val+1;
                if(val>sz)
                    break
                end
            end
            index=val-1;
        end


            
        % if(index<startpoint || index>endpoint)  % major error here. startpoint and endpoint are last updated.
        %    continue;
        % end
%        if(index==0)
 %           index=index+1;
  %      end
        

        if( abs ( posdata(index,1)-spikes{x}(timestamp)) > del_t*spike_window_tolerance )
            continue;
        end

        xx=posdata(index,2);
        yy=posdata(index,3);
        if(ignore_orig==1)
            if(xx==1 && yy==1)
                continue;
            end
        end
        xx=floor(xx);
        yy=floor(yy);
        if(xx==0)
            xx=1;
        end
        if(yy==0)
            yy=1;
        end
        
        frate(xx,yy)=frate(xx,yy)+1;
    end
    firingrates=[firingrates {frate}];
    %fprintf('Neuron %d complete\n',x);
    % waitbar(x/neurons,waitb,sprintf('Calculating firing rates... Cell %d/%d',x,neurons));
end


% Calculates firing rates from occupancy matrix -------------------------
for n=1:neurons
    for x=1:gridmax_x
        for y=1:gridmax_y
            if(firingrates{n}(x,y)~=0)
                firingrates{n}(x,y)=firingrates{n}(x,y)/(del_t*occupancy_matrix(x,y));
            end
        end
    end
end
% close(waitb);

intervals(1,:)=[];

params=[neurons; gridmax_x; gridmax_y; del_t];
model_params={params binsize_grid spatial_occ firingrates intervals occupancy_matrix vel1};
fprintf('DONE.\n');
end

