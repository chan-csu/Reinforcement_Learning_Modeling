Number_of_Models=10;
Models(1)=readCbModel('iJO1366.mat');
[BG,num_Models]=size(Models);
for i=1:num_Models
    Models(i).mets = regexprep(Models(i).mets, '_([a-zA-Z0-9])$', '\[$1\]');
    Models(i).rxns = regexprep(Models(i).rxns, '^EX_(\w+)_e$', 'EX_$1\(e\)');
end
Models(2)=addAmylaseToModel(Models(1));
for i=1:Number_of_Models-2
    Models(i+2)=Models(2);
end
% Models(3)=Models(2);
% Exchange_rxns(Models);
% Number_of_ODEs=length(Models)+length(Exchange_rxns(Models))+6;
Number_of_ODEs=5*length(Models)+4;

fprintf('The number of species in the reactor is: %d ',Number_of_ODEs)

Initial_cond=input('\nPlease input the Initial Conditions:\n');
if length(Initial_cond)~=Number_of_ODEs
    disp('Number of initial conditions is not equal to the number of "species" inside the reactor!\n');
    return
end
Inlet_cond=input('Please input the Inlet Conditions:\n');

if length(Inlet_cond)~=Number_of_ODEs
    disp('Number of inlet conditions is not equal to the number of "species" inside the reactor!');
    return
end
changeCobraSolver('ibm_cplex','all')
dil_rate=0.1;

global Data

for i=1:Number_of_Models
    Temp=find(strcmp(Models(i).rxns,'EX_amylase(e)'));
    Data(i).BiomassID=find(strcmp(Models(i).rxns,'BIOMASS_Ec_iJO1366_core_53p95M'));
             model=changeObjective(Models(i),'EX_glc__D(e)');
             model=changeRxnBounds(model,'EX_glc__D(e)',-20,'l');
             model=changeRxnBounds(model,'EX_glc__D(e)',0,'u');
             Temp_Sol(i).Res=optimizeCbModel(model,'Max','one');
             Min_Glc_Uptake=Temp_Sol(i).Res.x(find(cellfun(@isempty, strfind(Models(i).rxns,'EX_glc__D(e)'))==0));
             
    Data(i).Min_Glc_Uptake=Min_Glc_Uptake;       
    if Temp
        Data(i).Amylase=1;
        Data(i).Amylase_Ind=Temp;
    else
        Data(i).Amylase=0;
        Data(i).Amylase_Ind=0;
    end
end


c=HeterodFBA(Models,Initial_cond,Inlet_cond,dil_rate)
dbstop if error
function [Concentration]=HeterodFBA(Models,Initial_cond,Inlet_cond,dil_rate,death)
tic
t=[0,600]
EHM=Models(2);
EHM=changeRxnBounds(EHM,'EX_glc__D(e)',-20,'l');
EHS=optimizeCbModel(EHM,'max','one');
EHS.f=0;
params.Zero_Growth_Soln=EHS;
params.Inlet_cond=Inlet_cond;
params.Models=Models;
params.dil_rate=dil_rate;
Step=0.1;
[time,Concentration]=odeFwdEuler(@Sys,t,Initial_cond,Step,[],params);
[Row,Num]=size(Models);
hold on
subplot(3,1,1)
for i=1:Num
Org_Inds(i)=i*5-4;
end
plot(0:Step:t(2),Concentration(Org_Inds,:))
xlabel('Time(h)')
ylabel('Microbial-Concentration(Micrograms(ug/L)')
for i=1:Num
   Leg{i}= strcat('Org ' , num2str(i))
end
legend(Leg)

subplot(3,1,2)
plot(0:Step:t(2),Concentration(Num*5+1,:))
xlabel('Time(h)')
ylabel('Glucose(mmol/L)')

subplot(3,1,3)
plot(0:Step:t(2),Concentration(Num*5+2,:))
xlabel('Time(h)')
ylabel('Starch(g/L)')

toc
end

function [dcdt]=Sys(t,C,params)
global Data;
K=[0.1,0.1,0.1,0.000001];
Zero_Growth_Soln=params.Zero_Growth_Soln;
Models=params.Models;
Inlet_cond=params.Inlet_cond;
Dil_rate=params.dil_rate;
EX_RXNS=Exchange_rxns(Models);
[BG,num_Models]=size(Models);
j=0.01;
%% Setting the Boundary Conditions
% If a model is supposed to be Amylaze producer, In addition to glucose
% uptake rate constraint, it should be forced to produce amylaze with a
% rate calculated by Amylase_prod_rate function
MALT=1;
for i =1: num_Models
    
    if Data(i).Amylase
    j=j*3;
    APR=Amylase_prod_rate(C(i*5-3),C(i*5-2),MALT);
    Models(i)=changeRxnBounds(Models(i),'EX_amylase(e)',j*APR,'b');
    end
    
    Glc_uptake_rate(i)=-Glc_uptake_calc(C(i*5-3));
    Models(i)=changeRxnBounds(Models(i),'EX_glc__D(e)',Glc_uptake_rate(i),'b');
%     Models(i)=changeObjective(Models(i),'BIOMASS_Ec_iJO1366_core_53p95M');
    Sol(i).Res=optimizeCbModel(Models(i),'max','one');
    Glc_Treshold_Limit=0;
    if isnan(Sol(i).Res.f)==0 && C(i*5-3)>Glc_Treshold_Limit
        Sol(i).Mode=0;
        soln(i)=Sol(i).Res;
    else
        if Data(i).Amylase                                                                   % Some biological insight can be very useful here. For example, if amylase producer can not produce amylase                               % then what it does? just optimizes for biomass as the other one? remains stand by?
% 
        Models(i)=changeRxnBounds(Models(i),'EX_amylase(e)',0,'l');        % One question about APR: a system with very high starch and very low glucose, can not grow because the kinetics predicts


        end
%         Sol(i).Res=optimizeCbModel(Models(i),'max','one');
%                 if isnan(Sol(i).Res.f)==0
%                 Sol(i).Mode=1;
%                 soln(i)=Sol(i).Res;
%                 else
%                     Sol(i).Mode=2;    
%                     soln(i)=Zero_Growth_Soln;
%                     soln(i).x=zeros(size(Models(i).rxns));
%                     Glc_uptake_rate(i)=0;
%                 end
%                                                                        % very high amylase production and the low glucose can not support growth!
%         else

             Models(i)=changeObjective(Models(i),'EX_glc__D(e)');
             Models(i)=changeRxnBounds(Models(i),'EX_glc__D(e)',-20,'l');
             Models(i)=changeRxnBounds(Models(i),'EX_glc__D(e)',0,'u');
             Sol(i).Res=optimizeCbModel(Models(i),'Max','one');
             if (Glc_uptake_rate(i)<Data(i).Min_Glc_Uptake)
                Sol(i).Mode=1;
                soln(i)=Sol(i).Res;
             Glc_uptake_rate(i)=Data(i).Min_Glc_Uptake;
             else 
            Sol(i).Mode=2;    
            soln(i)=Zero_Growth_Soln;
            soln(i).x=zeros(size(Models(i).rxns));
            Glc_uptake_rate(i)=0;     
             end
       
   
             
    end
                if Data(i).Amylase
        Amyl=Data(i).Amylase_Ind;
        APRB(i)=soln(i).x(Amyl); 
                else
                   APRB(i)=0; 
              end            

    Death=Death_func(Models);
    if Sol(i).Mode==0 || Sol (i).Mode==1
    dcdt(i*5-4)=Dil_rate*(Inlet_cond(i*5-4)-C(i*5-4))+C(i*5-4)*soln(i).x(Data(i).BiomassID);
    else    
    dcdt(i*5-4)=Dil_rate*(Inlet_cond(i*5-4)-C(i*5-4));
    end
    N=12000;
    X=1000;

Starch_Breakdown_Rate=Glc_from_Starch(C(i*5-2),C(i*5))+Dextrine_From_Starch(C(i*5-2),C(i*5))+Maltose_From_Starch(C(i*5-2),C(i*5))+Maltotriose_From_Starch(C(i*5-2),C(i*5));

    dcdt(i*5-3)=K(1)*(C(num_Models*5+1)-C(i*5-3))+N*Glc_from_Starch(C(i*5-2),C(i*5))+N/X*Glc_from_Dextrin(C(i*5-1),C(i*5))+C(i*5-4)*Glc_uptake_rate(i);
    dcdt(i*5-2)=K(2)*(C(num_Models*5+2)-C(i*5-2))-Starch_Breakdown_Rate;
    dcdt(i*5-1)=K(3)*(C(num_Models*5+3)-C(i*5-1))+X*Dextrine_From_Starch(C(i*5-2),C(i*5))-Maltose_From_Dextrine(C(i*5-1),C(i*5))-Maltotriose_From_Dextrine(C(i*5-1),C(i*5))-Glc_from_Dextrin(C(i*5-1),C(i*5));
    dcdt(i*5)=K(4)*(C(num_Models*5+4)-C(i*5))+APRB(i)*C(i*5-4);
    Transfer(i,1)=K(1)*(C(num_Models*5+1)-C(i*5-3));
    Transfer(i,2)=K(2)*(C(num_Models*5+2)-C(i*5-2));
    Transfer(i,3)=K(3)*(C(num_Models*5+3)-C(i*5-1));
    Transfer(i,4)=K(4)*(C(num_Models*5+4)-C(i*5));
    
end

Starch_Breakdown_Rate=Glc_from_Starch(C(num_Models*5+2),C(num_Models*5+4))+Dextrine_From_Starch(C(num_Models*5+2),C(num_Models*5+4))+Maltose_From_Starch(C(num_Models*5+2),C(num_Models*5+4))+Maltotriose_From_Starch(C(num_Models*5+2),C(num_Models*5+4));

Transfer=-sum(Transfer,1)
dcdt(num_Models*5+1)=Transfer(1,1)+Dil_rate*(Inlet_cond(num_Models*5+1)-C(num_Models*5+1))+N*Glc_from_Starch(C(num_Models*5+2),C(num_Models*5+4))+N/X*Glc_from_Dextrin(C(num_Models*5+3),C(num_Models*5+4));
dcdt(num_Models*5+2)=Transfer(1,2)+Dil_rate*(Inlet_cond(num_Models*5+2)-C(num_Models*5+2))-Starch_Breakdown_Rate;
dcdt(num_Models*5+3)=Transfer(1,3)+Dil_rate*(Inlet_cond(num_Models*5+3)-C(num_Models*5+3))+X*Dextrine_From_Starch(C(num_Models*5+2),C(num_Models*5+4))-Maltose_From_Dextrine(C(num_Models*5+3),C(num_Models*5+4))-Maltotriose_From_Dextrine(C(num_Models*5+3),C(num_Models*5+4))-Glc_from_Dextrin(C(num_Models*5+3),C(num_Models*5+4));    
dcdt(num_Models*5+4)=Transfer(1,4)+Dil_rate*(Inlet_cond(num_Models*5+4)-C(num_Models*5+4))

% %% Solving the LPs
% %At this part using the constraints from previous part, the optimization
% %problem is solved for each model.
% 
% for i =1: num_Models
% 
%     
% 
% %%
% 
% Total_gluc_Uptake=0;
% 
% for i=1:num_Models
%     Glc_Up_Id=findRxnIDs(Models(i),'EX_glc__D(e)');
%     Total_gluc_Uptake=Total_gluc_Uptake+C(i+6)*soln(i).x(Glc_Up_Id);
% end
% 
% Amylase_Production=0;
% 
% for i=1:(num_Models)
%     if Data(i).Amylase
%         Amyl=Data(i).Amylase_Ind;
%         Amylase_Production=Amylase_Production+C(i+6)*soln(i).x(Amyl);     
%     end
% end
% 
% Starch_Breakdown_Rate=Glc_from_Starch(C(2),C(6))+Dextrine_From_Starch(C(2),C(6))+Maltose_From_Starch(C(2),C(6))+Maltotriose_From_Starch(C(2),C(6));
% 
% dcdt(1)=Dil_rate*(Inlet_cond(1)-C(1))+N*Glc_from_Starch(C(2),C(6))+N/X*Glc_from_Dextrin(C(5),C(6))+Total_gluc_Uptake;
% dcdt(2)=Dil_rate*(Inlet_cond(2)-C(2))-Starch_Breakdown_Rate;
% dcdt(3)=Dil_rate*(Inlet_cond(3)-C(3))+N/2*Maltose_From_Starch(C(2),C(6))+N/(2*X)*Maltose_From_Dextrine(C(5),C(6));
% dcdt(4)=Dil_rate*(Inlet_cond(4)-C(4))+N/3*Maltotriose_From_Starch(C(2),C(6))+N/(3*X)*Maltotriose_From_Dextrine(C(5),C(6));
% dcdt(5)=Dil_rate*(Inlet_cond(5)-C(5))+X*Dextrine_From_Starch(C(2),C(6))-Maltose_From_Dextrine(C(5),C(6))-Maltotriose_From_Dextrine(C(5),C(6))-Glc_from_Dextrin(C(5),C(6));
% dcdt(6)=Dil_rate*(Inlet_cond(6)-C(6))+Amylase_Production-C(6)*0.1;
% 
% for i=1:num_Models
%     Death=Death_func(Models);
%     if Sol(i).Mode==0 || Sol (i).Mode==1
%     dcdt(i+6)=Dil_rate*(Inlet_cond(i+6)-C(i+6))+C(i+6)*soln(i).f;
%     else    
%     dcdt(i+6)=Dil_rate*(Inlet_cond(i+6)-C(i+6))-C(i+6)*(Death(i));
%     end
% end
 

% for i=1:(length(EX_RXNS))
% %     dcdt(1+length(num_Models)+i)=Dil_rate*(Inlet_cond(1+length(num_Models)+i)-C(1+length(num_Models)+i));
%     dcdt(6+num_Models+i)=Dil_rate*(Inlet_cond(6+length(num_Models)+i)-C(6+length(num_Models)+i));    
%     for j=1:num_Models
%             if find(cellfun(@isempty, strfind(Models(j).rxns,EX_RXNS(i)))==0)
%             ind=find(cellfun(@isempty, strfind(Models(j).rxns,EX_RXNS(i)))==0);
%                if isempty(soln(j).x)==0                  
%                    dcdt(6+num_Models+i)=dcdt(6+(num_Models)+i)+C(j+1)*soln(j).x(ind);
%                end
%             end
%         end
% end



dcdt=dcdt';

for i=1:num_Models
Rows{i*5-4}=strcat('Model ',num2str(i));
% fprintf('The mode of model %d is: %d ',[i Sol(i).Mode])

Rows{i*5-3}='Glucose';
Rows{i*5-2}='Starch';
Rows{i*5-1}='Dextrine';
Rows{i*5}='Amylase';
end
Rows{num_Models*5+1}='Free_Glucose';
Rows{num_Models*5+2}='Free_Starch';
Rows{num_Models*5+3}='Free_Dextrine';
Rows{num_Models*5+4}='Free_Amylase';

table(Rows',C,dcdt)
end 

function rate=Glc_uptake_calc(C)
Vmax=1.5;
Km=2.5;
rate=Vmax*C/(Km+C);
end

function Ex_rxns=Exchange_rxns(Models)
[BG,num_Models]=size(Models);
Ex_rxns={}

for i=1:num_Models
    exchanges=find(cellfun(@isempty, strfind(Models(i).rxns,'EX_'))==0);
    Ex_rxns=union(Ex_rxns,Models(i).rxns(exchanges));
end

Ex_rxns(find(cellfun(@isempty, strfind(Ex_rxns,'EX_amylase(e)'))==0))=[];
Ex_rxns(find(cellfun(@isempty, strfind(Ex_rxns,'EX_glc__D(e)'))==0))=[];

 end

function GFS=Glc_from_Starch(Cs,CA)
%From Brandam et al. 
T=310; %temperature in K
k_gl=0.00023;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
GFS=k_gl*a_alpha*Cs;
end

function GFD=Glc_from_Dextrin(Cd,CA)
kp_gl=2.9*10^(-12);
T=310;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
GFD=kp_gl*a_alpha*Cd;
end

function DFS=Dextrine_From_Starch(Cs,CA)
k_dex=0.00000317;
T=310;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
DFS=k_dex*a_alpha*Cs;
end

function MFS=Maltose_From_Starch(Cs,CA)
k_mal=0.00389;
T=310;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
MFS=k_mal*a_alpha*Cs;
end 

function MTFS=Maltotriose_From_Starch(Cs,CA)
k_mlt=0.00117;
T=310;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
MTFS=k_mlt*a_alpha*Cs;
end

function MFD=Maltose_From_Dextrine(Cd,CA)
kp_mat=1.2*10^(-10);
T=310;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
MFD=kp_mat*a_alpha*Cd;
end

function MTFD=Maltotriose_From_Dextrine(Cd,CA)
kp_mlt=1.5*10^(-10);
T=310;
a_alpha=CA*0.022*(-0.0011*T^3+1.091*T^2-352.89*T+38008.3);
MTFD=kp_mlt*a_alpha*Cd;
end

function APR=Amylase_prod_rate(C_glc,C_starch,mal)
% enzyme specific production rate inhibited by glcuose and maltose
% Yoo et al., 1988. Biotech Bioeng
mal=1;
Ki_g = 35.1751;  % mM
n_g = 4.255;  % unitless

Ki_m = 22.5066; % mM
n_m = 5.222; % unitless

Ka_starch = 1;  % g/L
 

% Km / [S]. S should just be the amino acids and E is the ribosome
% value unknown. Just treated as a scaling factor for Vmax. Using 1 is fine
KmOverS = 1;

% Vmax here is kcat multiplied by ribosome concentration g/hr/gDW ~40 unit/mgDW/hr

% mal=1 %for sake of simplicity!
    % Catalytic Hill's inhibition and activation
APR = 1./ (KmOverS + Ka_starch / C_starch + (C_glc / Ki_g) .^ n_g + (mal / Ki_m) .^ n_m + 1);

% % [drdStarch, drdGlucose, drdMaltose]
% drdS = @(starch, glc, mal, Vmax) (Vmax / ((KmOverS + (glc / Ki_g) .^ n_g + (mal / Ki_m) .^ n_m + 1) * starch + Ka_starch) ^ 2) ...
% * [Ka_starch, ...
% starch ^ 2 * n_g * glc ^ (n_g- 1) / (Ki_g ^ n_g), ...
% starch ^ 2 * n_m * mal ^ (n_m - 1) / (Ki_m ^ n_m)];

end

function Death=Death_func(Models)
%TBD: This function can be even extended to include temperature and pH
%dependencies.
[BG,num_Models]=size(Models);
Death=0.05*ones(num_Models);
end




