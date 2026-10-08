classdef MOKPR < PROBLEM
% <1999> <multi/many> <binary> <large/none>
% Zitzler-Thiele multi-objective 0/1 knapsack, EXACT thesis instances,
% with PlatEMO's OWN greedy repair instead of explicit constraints.
%
% Why this variant exists
% -----------------------
% MOKPX states the capacity as an explicit constraint (CalCon), following
% PlatEMO 4's own MOKP. Under that formulation an algorithm must implement
% constraint handling, and every PlatEMO algorithm that does not declare
% <constrained> returns only infeasible solutions (verified: EAG-MOEA/D,
% 0 feasible solutions in 2 runs).
%
% PlatEMO 3.5's MOKP stated the SAME problem differently: the capacity was
% enforced by a greedy repair inside the problem (CalDec), so every algorithm
% was automatically feasible. The repair below is that function, copied
% verbatim from
%   PlatEMO-3.5/PlatEMO/Problems/Multi-objective optimization/Real-world MOPs/MOKP.m
% with one substitution: the shipped C = sum(W,2)/2 is replaced by the exact
% per-instance capacities of the thesis (identical on 7 of the 9 instances,
% a few units apart on 500.3 and 750.4).
%
% This is the platform's own repair operator, not a repair written for this
% study -- the distinction that invalidated the earlier pymoo comparison.
% Objectives, data and budget are otherwise identical to MOKPX.

%------------------------------- Reference --------------------------------
% E. Zitzler and L. Thiele. Multiobjective evolutionary algorithms: A
% comparative case study and the strength Pareto approach. IEEE
% Transactions on Evolutionary Computation, 1999, 3(4): 257-271.
%--------------------------------------------------------------------------

    properties(SetAccess = private)
        P;  % [M x D] profit of each item according to each knapsack
        W;  % [M x D] weight of each item according to each knapsack
        C;  % [1 x M] EXACT capacity of each knapsack (from instance file)
    end
    methods
        function Setting(obj)
            if isempty(obj.M); obj.M = 2; end
            if isempty(obj.D); obj.D = 250; end
            obj.encoding = 4 + zeros(1,obj.D);
            file = sprintf('MOKPX-M%d-D%d.mat',obj.M,obj.D);
            file = fullfile(fileparts(mfilename('fullpath')),file);
            data = load(file,'P','W','C');
            obj.P = data.P;
            obj.W = data.W;
            obj.C = data.C(:);
        end
        %% Repair invalid solutions (PlatEMO 3.5 MOKP.CalDec, verbatim,
        %% with the exact capacities substituted for sum(W,2)/2)
        function PopDec = CalDec(obj,PopDec)
            C = obj.C;
            [~,rank] = sort(max(obj.P./obj.W));
            for i = 1 : size(PopDec,1)
                while any(obj.W*PopDec(i,:)'>C)
                    k = find(PopDec(i,rank),1);
                    PopDec(i,rank(k)) = 0;
                end
            end
        end
        %% Calculate objective values (PlatEMO minimizes: total profit - achieved)
        function PopObj = CalObj(obj,PopDec)
            PopObj = repmat(sum(obj.P,2)',size(PopDec,1),1) - PopDec*obj.P';
        end
        %% Generate a point for hypervolume calculation
        function R = GetOptimum(obj,~)
            R = sum(obj.P,2)';
        end
    end
end
