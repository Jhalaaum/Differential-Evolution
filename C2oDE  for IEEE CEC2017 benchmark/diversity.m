function  [p,objF,conV,FES]=diversity(p,objF,conV,minVar,maxVar,problem,FES)
global count;
[popsize,n]=size(p);

    if std(conV)<1.e-8 && isempty(find(conV==0))
        p(1:popsize,:)=repmat(minVar,popsize,1)+rand(popsize,n).*repmat((maxVar-minVar),popsize,1);
        [objF, conV]=get_Fitness(p, problem);
        FES=FES+popsize;
        count= count+1;
    end
    % numStagnated = sum(conV>0);
end
