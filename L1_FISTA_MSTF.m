function [beta_opt,fun] = L1_FISTA_MSTF(X,y,lambda,opts,n)


%% initilziation


if isfield(opts, 'init_beta')
    beta_current=opts.init_beta;
    beta_old = opts.init_beta;    
else    
    beta_current = zeros(size(X{1},2),1);
    beta_old = beta_current;
end

T = length(y);


t=1;
t_old=0;

iter=0;
fun=[];

L = 1;

is_contin=1;

%% main loop
while iter<opts.max_iter & is_contin
    alpha = (t_old-1)/t;
    beta_s = (1+alpha)*beta_current - alpha*beta_old;
    grad = grad_f(beta_s);
    beta_old =beta_current;
    while 1
        beta_current = proximalL1norm(beta_s - grad/L, lambda/L); 
        if eval_f(beta_current) <= eval_f(beta_s) + dot(grad,beta_current-beta_s) + ...
                            (norm(beta_current-beta_s,2)).^2/(2/L)
           break;
        else
            L = L*2;
        end 
    end

    fun = cat(1,fun, eval_f(beta_current) + lambda*norm(beta_current,1));
    

    if iter>=2 & abs(fun(end-1) - fun(end)) <=opts.rel_tol * fun(end-1)
        break;
    end
    
    iter=iter+1;
    t_old=t;
    t=0.5 * (1+(1+4*t^2)^0.5);

end
beta_opt = beta_current;

%% private function

    function fun=eval_f(beta)
        fun = 0;
        for t=1:T
            fun = fun + norm(y{t} -  X{t} * beta,2)^2 /(2*n(t));
        end
    end

    function grad = grad_f(beta)
        grad = 0;
        for t=1:T
            grad = grad + X{t}'*(X{t}*beta-y{t})/n(t);
        end
    end

end

