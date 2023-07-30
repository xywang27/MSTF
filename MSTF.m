function [W_temp,a,C,F,G_temp,fun] = MSTF(X_cell,y_cell,hyp, opts)
%X_cell is a T*1 cell containing data.
%X_cell{i} means ith task data which is n_i*1
%X_cell{i}{n} means ith task nth data point which is a (d_1+1)*(d_2+1)*...*(d_V+1) tensor
%y_cell is a T*1 cell containing label
%y_cell{i} means ith task lebel which is n_i*1
%W_temp is T*1 cell, W_temp{t} is the weight tensor of task t which is (d_1+1)*(d_2+1)*...*(d_V+1)

%% Initialization 
% hyp =[lambda1, lambda2,lambda3,alpha]

T = length(y_cell);
V = length(size(X_cell{1}{1}));
d = zeros(V,1);
r = zeros(V,1);
pidiplus1 = 1;
n = zeros(T,1);
alpha = hyp(4);
for i = 1:V
    d(i) = size(X_cell{1}{1},i) - 1;
    r(i) = max(floor(alpha * (d(i)+1)),1);
    pidiplus1 = pidiplus1*size(X_cell{1}{1},i);
end
for i = 1:T
    n(i) = length(X_cell{i});
end
C = cell(V,1);
a = cell(V,1);
F = cell(V,1);
G_temp = cell(T,1); %G_t
W_temp = cell(T,1); %W_t
rng('default')
for v = 1:V
    C{v} = normrnd(0,sqrt(2),[d(v)+1,r(v)]);
    a{v} = ones(d(v)+1,1);
    F{v} = diag(a{v})*C{v};
end
for i = 1:T
    G_temp{i} = normrnd(0,sqrt(2),r');
end

G_transformed = ones(prod(r),T);
exponent1 = 1/(opts.p+ opts.k);

X_cell_transformed = cell(T,1);
for t = 1:T
    X_cell_transformed{t} = zeros(n(t),pidiplus1);
    for i = 1:n(t)
        X_cell_transformed{t}(i,:) = reshape(X_cell{t}{i},[1,pidiplus1]);
    end
end

X_cell_transformed_forC = cell(V,1);
for v = 1:V
    X_cell_transformed_forC{v} = cell(T,1);
    for t = 1:T
        X_cell_transformed_forC{v}{t} = zeros(n(t),pidiplus1);
        for i = 1:n(t)
            X_cell_transformed_forC{v}{t}(i,:) = reshape(double(tenmat(X_cell{t}{i},v)),[1,pidiplus1]);
        end
    end
end


fun =[];
%% main
tic
iter=0;
while iter<opts.max_iter

    % update C_v and a_v
    for v = 1:V
        X_cell_bar_transformed = cell(T,1);
        Fminusv = 1;
        for i = V:-1:1
            if i ~= v
                Fminusv = kron(Fminusv,F{i});
            end
        end
        for t = 1:T
            X_cell_bar_transformed{t} = zeros(n(t),pidiplus1);
            X_cell_bar_transformed{t} = X_cell_transformed_forC{v}{t}*kron(double(Fminusv*(tenmat(G_temp{t},v))'),diag(a{v}));
        end
        switch opts.p
            case 1                
                [Cv_transformed,b_fun] = L1_FISTA_MSTF(X_cell_bar_transformed,y_cell, hyp(2), opts, n); 
            case 2
                left = 2 * hyp(2) * eye(size(X_cell_bar_transformed{1},2));
                right = 0;
                for t = 1:T
                    left = left +  X_cell_bar_transformed{t}'*X_cell_bar_transformed{t}/size(X_cell_bar_transformed{t},1) ;
                    right = right + X_cell_bar_transformed{t}'*y_cell{t}/size(X_cell_bar_transformed{t},1);
                end
                Cv_transformed =left\right;
            otherwise
                disp('other value')
        end
        C{v} = reshape(Cv_transformed,[d(v)+1,r(v)]);
        F_temp = diag(a{v})*C{v};
        for j = 1: d(v)+1
            switch opts.p
                case 1
                    temp = norm(F_temp(j,:),1);
                case 2
                    temp = sum(F_temp(j,:).^2);
            end
            a{v}(j,1) = ((hyp(2)/hyp(1))*temp)^exponent1;
        end
        F{v} = diag(a{v})*C{v};
    end
    %update G_t
    X_cell_bar_transformed = [];
    FVto1_bar = 1;
    for i = V:-1:1
        FVto1_bar = kron(FVto1_bar,F{i});
    end
    for t=1:T
        X_cell_bar_transformed = X_cell_transformed{t}*FVto1_bar;
        y_cell_bar = y_cell{t};
        left = X_cell_bar_transformed'*X_cell_bar_transformed/size(X_cell_bar_transformed,1) + 2 * hyp(3) * eye(size(X_cell_bar_transformed,2));
        right = X_cell_bar_transformed'*y_cell_bar/size(X_cell_bar_transformed,1);
        G_transformed(:,t) =left\right;
        G_temp{t} = reshape(G_transformed(:,t),r');
    end
    
    
    for t = 1:T
        W_temp{t} = ttm(tensor(G_temp{t}),{F{1},F{2}},[1:V]);
%         W_temp{t} = ttm(tensor(G_temp{t}),{F{1},F{2},F{3}},[1:V]);
%         W_temp{t} = ttm(tensor(G_temp{t}),{F{1},F{2},F{3},F{4}},[1:V]);
%         W_temp{t} = ttm(tensor(G_temp{t}),{F{1},F{2},F{3},F{4},F{5}},[1:V]);
%         W_temp{t} = ttm(tensor(G_temp{t}),{F{1},F{2},F{3},F{4},F{5},F{6}},[1:V]);
    end
    fun = cat(1,fun,Obj_multiplicative(a,C,G_temp,W_temp));
    
    if iter>=2 && abs(fun(end)-fun(end-1))<= opts.rel_tol*fun(end-1)
        break;
    end
    % stopping criteria       
    iter=iter+1;

end



    %%
    
    function val = Obj_multiplicative(a,C,G,W)
        
        val = 0;
        
        for v = 1:V
            val = val + hyp(1) * norm(a{v},opts.k)^opts.k;
            for t2= 1:r(v)
                val = val + hyp(2) * norm(C{v}(:,t2),opts.p)^opts.p;
            end 
        end
        
        for t3= 1:T
            val = val + hyp(3) * norm(tensor(G{t3}))^2;
        end 
        
        
        for t4=1:T
            val = val + norm(y_cell{t4} -  X_cell_transformed{t4} * double(reshape(W{t4},[pidiplus1,1])))^2 /(2*n(t4));
        end             
    end


end

