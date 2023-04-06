clc
clear

data=readtable('accepted_2007_to_2018Q4.csv');

%%

N_max=2260701;                                                             % total number of data points

edge_loan=0:10000:40000;                                                   % bands for discretizing loan amount
edge_FICO=[0,650:50:850];                                                  % bands for discretizing credit rating
edge_rate=5:5:35;                                                          % bands for discretizing interest rates

X=zeros(N_max,3);

X(:,1)=discretize(data(1:N_max,:).loan_amnt,edge_loan);
X(:,2)=discretize(data(1:N_max,:).fico_range_high,edge_FICO);
X(:,3)=discretize(data(1:N_max,:).int_rate,edge_rate);

%%

[I,J]=find(isnan(X));                                                      % eliminating data rows that contain invalid entries
X(I,:)=[];

%%

save('discretized_data','X')                                               % saving the formatted data for future use

%%

P=length(edge_loan)*length(edge_FICO)*length(edge_rate);                   % cardinality of set \Xi 

Xi=zeros(3,P);                                                             % constructing set \Xi
index=0;                                                                   
for j2=1:length(edge_loan)
    for j3=1:length(edge_FICO)
        for j4=1:length(edge_rate)
            index=index+1;
            Xi(:,index)=[j2;j3;j4];
        end
    end
end

%%

Delta=0;                                                                   % computing sensitivity for differential privacy
for i=1:P
    for j=1:P
        Delta=max([Delta,norm(Xi(:,i)-Xi(:,j),inf)]);
    end
end

%%
clc

N=[1e4 3e4 1e5 3e5 1e6];                                                   % the range we vary N for experiments
epsilon=[3, 10, 30, 100];                                                  % the range we vary \epsilon for experiments

X_test=X(1000001:length(X),:);                                             % use up to the first million for learning and the rest for testing

it_max=10;

e_DRO  =zeros(length(N),length(epsilon));
e_noisy=zeros(length(N),length(epsilon));
e_clean=zeros(length(N),1);

for it=1:it_max
    
    X=X(randperm(length(X)),:);

    for iEpsilon=1:length(epsilon)
        %%
        
        O=zeros(P,P);                                                      % computing conditional probability for realizing differentially-private data
        for i=1:P
            for j=1:P
                O(j,i)=exp(-epsilon(iEpsilon)*norm(Xi(:,i)-Xi(:,j),inf)/2/Delta);
            end
        end
        for i=1:P
            O(:,i)=O(:,i)/sum(O(:,i));
        end

        %%

        XX=zeros(max(N),3);
        for i=1:max(N)
            for j=1:P
                if isequal(Xi(:,j),X(i,:)')
                    temp=j;
                    break
                end
            end
            XX(i,:)=Xi(:,sample(O(:,temp)'))';                             % generating differentially-private data based on the developed conditional probability
        end
        
        %%
        
        for iN=1:length(N)
            
            fprintf('N=%d | epsilon=%d | iteration number=%d\n', ...
                    [N(iN), epsilon(iEpsilon), it])                        % to track progress of the experiments
            
            %%

            hatP_prime=zeros(1,P);                                         % empirical probability of the noisy data
            for i=1:N(iN)
                for j=1:P
                    if isequal(Xi(:,j),XX(i,:)')
                        temp=j;
                        break
                    end
                end
                hatP_prime(temp)=hatP_prime(temp)+1/N(iN);
            end

            %%
            
            alpha=1/N(iN)^1.1;
            var_epsilon=sqrt(max([P,2*log(2/alpha)])/N(iN));               % radius of the ambiguity set

            cvx_solver SeDuMi                                              % Disributionally robust optimization using noisy data by convex optimization
            cvx_begin 
                variable l(1,P) 
                variable m(1,P) 
                variable r 
                variable t 
                variable x(1,3)
                minimize( r+2*var_epsilon*t+sum((m-l).*hatP_prime) )
                subject to
                    for i=1:P
                        m(i)+l(i)<=t;
                        m(i)>=0;
                        l(i)>=0; 
                        (x*[Xi(1:2,i); 1]-Xi(3,i)).^2+sum((l-m).*(O(:,i)'))<=r;
                    end
            cvx_end
            
            e_DRO(iN,iEpsilon)=e_DRO(iN,iEpsilon)+...
                    mean((x*[X_test(:,1:2)'; ...
                    ones(1,length(X_test))]-X_test(:,3)').^2)/it_max;      % computing regression error for DRO

            %%
            
            xdata=[XX(1:N(iN),1:2) ones(N(iN),1)];
            ydata= XX(1:N(iN),3);
            beta_noisy=xdata\ydata;                                        % optimal linear regression with noisy data (naive regression)
            
            e_noisy(iN,iEpsilon)=e_noisy(iN,iEpsilon)+ ...
                    mean((beta_noisy'*[X_test(:,1:2)'; ...
                    ones(1,length(X_test))]-X_test(:,3)').^2)/it_max;      % computing regression error for naive regression
                  
        end
    end

    for iN=1:length(N)
        xdata=[X(1:N(iN),1:2) ones(N(iN),1)];
        ydata= X(1:N(iN),3);
        beta_clean=xdata\ydata;                                            % optimal linear regression with noiseless data (the best achievable performance)

        e_clean(iN)=e_clean(iN)+...
            mean((beta_clean'*[X_test(:,1:2)'; ...
            ones(1,length(X_test))]-X_test(:,3)').^2)/it_max;              % computing regression error for noiseless optimal regression
    end
    
end

%%

save('errors_DRO_vs_naive','e_DRO','e_noisy','e_clean')

%%

C=[0.0000, 0.4470, 0.7410; ...
   0.8500, 0.3250, 0.0980; ...
   0.9290, 0.6940, 0.1250; ...
   0.4940, 0.1840, 0.5560; ...
   0.4660, 0.6740, 0.1880; ...
   0.3010, 0.7450, 0.9330; ...
   0.6350, 0.0780, 0.1840];

figure;

l=cell(9,1);
    
for iEpsilon=1:length(epsilon)
    
    semilogx(N,e_DRO(:,iEpsilon)  ,'-o' ,'linewidth',2,'color',C(iEpsilon,:))
     
    l{2*iEpsilon-1}="DRO" + " " + "$\epsilon=" + sprintf('%0.1f',epsilon(iEpsilon)) + "$";
    
    hold on
    
    semilogx(N,e_noisy(:,iEpsilon),'-*','linewidth',2,'color',C(iEpsilon,:))
    
    l{2*iEpsilon}="Naive" + " " + "$\epsilon=" + sprintf('%0.1f',epsilon(iEpsilon)) + "$";
    
end

semilogx(N,10*e_clean,'-k','linewidth',2)

l{9}="Non-private";


leg1=legend(l);
set(leg1,'Interpreter','latex');
set(leg1,'FontSize',17,'Location','northoutside');
set(gca,'FontSize',17)
leg1.NumColumns=4;

% 
ylim([0 5])

%%

for iEpsilon=1:length(epsilon)

    figure;
        
    semilogx(N,e_DRO(:,iEpsilon)  ,'-o' ,'linewidth',2,'color',[0 0 0], 'MarkerSize',10)
     
    leg_text{1}="DRO";
    
    hold on
    
    semilogx(N,e_noisy(:,iEpsilon),'-d','linewidth',2,'color',[0 0 0], 'MarkerSize',10)
    
    leg_text{2}="Naive";
    
    hold on  
    
    semilogx(N,e_clean,'-r','linewidth',2)

    leg_text{3}="Non-private";  
    
    if iEpsilon==1
        ylim([0 6])
    else
        ylim([0 3])
    end
    
    grid on
    
    leg{iEpsilon}=legend(leg_text);
    set(leg{iEpsilon},'Interpreter','latex');
    set(leg{iEpsilon},'FontSize',17,'Location','northoutside');
    set(gca,'FontSize',17)
    leg{iEpsilon}.NumColumns=3;
    
    title("$\epsilon=" + sprintf('%0.1f',epsilon(iEpsilon)) + "$",'interpreter','latex')

    x0=600;
    y0=600;
    width=400;
    height=400;
    set(gcf,'position',[x0,y0,width,height])
    
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(gcf,strcat('test',sprintf('%d',iEpsilon),'.pdf'),'-dpdf','-r0')
    
end