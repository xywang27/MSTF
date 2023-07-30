function [W_transformed,A_transformed,B_transformed,fun] = vMSTF(X_cell,y_cell,hyp, opts)
%X_cell is a T*1 cell containing data.
%X_cell{i} means ith task data which is n_i*1
%X_cell{i}{n} means ith task nth data point which is a (d_1+1)*(d_2+1)*...*(d_V+1) tensor
%y_cell is a T*1 cell containing label
%y_cell{i} means ith task label which is n_i*1
%W_transformed is d*T, d = (d_1+1)*(d_2+1)*...*(d_V+1)

%% Initialization 
% hyp =[lambda1, lambda2]

T = length(y_cell);
V = length(size(X_cell{1}{1}));
d = zeros(V,1);
pidiplus1 = 1;
n = zeros(T,1);
for i = 1:V
    d(i) = size(X_cell{1}{1},i) - 1;
    pidiplus1 = pidiplus1*size(X_cell{1}{1},i);
end
for i = 1:T
    n(i) = length(X_cell{i});
end
A = ones(size(X_cell{1}{1}));
A_transformed = reshape(A,[pidiplus1,1]);
B_transformed = ones(pidiplus1,T);
exponent1 = 1/(opts.p+ opts.k);

X_cell_transformed = cell(T,1);
for t = 1:T
    X_cell_transformed{t} = zeros(n(t),pidiplus1);
    for i = 1:n(t)
        X_cell_transformed{t}(i,:) = reshape(X_cell{t}{i},[1,pidiplus1]); 
    end
end



fun =[];
%% main
tic
iter=0;
while iter<opts.max_iter
    
    % update b
    for t=1:T
        X_cell_bar_transformed = X_cell_transformed{t}*diag(A_transformed);
        y_cell_bar = y_cell{t};
        switch opts.p
            case 1                
                [B_transformed(:,t),b_fun] = L1_FISTA_vMSTF(X_cell_bar_transformed,y_cell_bar, hyp(2), opts); 
            case 2
                left = X_cell_bar_transformed'*X_cell_bar_transformed/size(X_cell_bar_transformed,1) + 2 * hyp(2) * eye(size(X_cell_bar_transformed,2));
                right = X_cell_bar_transformed'*y_cell_bar/size(X_cell_bar_transformed,1);
                B_transformed(:,t) =left\right;
            otherwise
                disp('other value')
        end                
    end
    W_transformed =diag(A_transformed)* B_transformed;
    
    % update a  
    for j=1:pidiplus1
        switch opts.p
            case 1
                temp = norm(W_transformed(j,:),1);
            case 2
                temp = sum(W_transformed(j,:).^2);
        end
        A_transformed(j,1) = ((hyp(2)/hyp(1))*temp)^exponent1;
    end    
    
    W_transformed = diag(A_transformed)*B_transformed;
    fun = cat(1,fun,Obj_multiplicative(A_transformed,B_transformed,W_transformed));
    
    if iter>=2 && abs(fun(end)-fun(end-1))<= opts.rel_tol*fun(end-1)
        break;
    end

    % stopping criteria       
    iter=iter+1;

end




    %%
    
    function val = Obj_multiplicative(a,B,W)
        
        val = hyp(1) * norm(a,opts.k)^opts.k;
        
        for t2= 1:T
            val = val + hyp(2) * norm(B(:,t2),opts.p)^opts.p;
        end 
        
        
        for t2=1:T
            val = val + norm(y_cell{t2} -  X_cell_transformed{t2} * W(:,t2))^2/(2*n(t2));
        end             
    end

end

