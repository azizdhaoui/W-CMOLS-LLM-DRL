classdef MOKPX < PROBLEM
% <1999> <multi/many> <binary> <large/none> <constrained>
% Zitzler-Thiele multi-objective 0/1 knapsack with EXACT thesis instances.
%
% Identical formulation to PlatEMO's built-in MOKP (Zitzler & Thiele 1999)
% with ONE data-level difference: P (profits), W (weights) AND the exact
% capacities C are loaded from MOKPX-M<M>-D<D>.mat, so the problem solved
% here is bit-for-bit the same 9 instances used everywhere in the thesis
% (PlatEMO's MOKP assumes C = sum(W,2)/2, which matches 7/9 of our
% instances but is off by a few units on 500.3 and 750.4 -> exact C).
%
% The .mat files are generated from the thesis's data/<inst>.txt by
% comparison/competitors/platemo (Python: scipy.io.savemat). No algorithm code is
% touched: this class only carries the instance data.

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
        %% Default settings of the problem
        function Setting(obj)
            if isempty(obj.M); obj.M = 2; end
            if isempty(obj.D); obj.D = 250; end
            obj.encoding = 4 + zeros(1,obj.D);
            file = sprintf('MOKPX-M%d-D%d.mat',obj.M,obj.D);
            file = fullfile(fileparts(mfilename('fullpath')),file);
            data = load(file,'P','W','C');
            obj.P = data.P;
            obj.W = data.W;
            obj.C = data.C(:)';
        end
        %% Calculate objective values (PlatEMO minimizes: total profit - achieved)
        function PopObj = CalObj(obj,PopDec)
            PopObj = repmat(sum(obj.P,2)',size(PopDec,1),1) - PopDec*obj.P';
        end
        %% Calculate constraint violations (all M knapsack capacities, EXACT C)
        function PopCon = CalCon(obj,PopDec)
            PopCon = PopDec*obj.W' - repmat(obj.C,size(PopDec,1),1);
        end
        %% Generate a point for hypervolume calculation
        function R = GetOptimum(obj,~)
            R = sum(obj.P,2)';
        end
    end
end
