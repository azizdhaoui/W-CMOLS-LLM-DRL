% run_platemo_extra.m — 6 ADDITIONAL published constrained MOEAs from PlatEMO,
% run AS-IS on the thesis's 9 exact Zitzler-Thiele MOKP instances.
%
% Compliance (encadrante's rule: consume algorithms unmodified, inject DATA only):
%   - every algorithm is taken straight from PlatEMO (BIMK), zero code changes
%   - PlatEMO handles the knapsack capacity constraints NATIVELY (each algorithm
%     has its own constraint-handling); we never write or inject a repair operator
%   - only the problem DATA enters, via MOKPX.m (P/W/exact capacities)
%   - all 6 are tagged <binary> <constrained> in their own PlatEMO headers, i.e.
%     they officially support this encoding
%
% Algorithms (ordered by priority — stop anytime, the earlier ones matter most):
%   MOEADDE   2009  H. Li, Q. Zhang — MOEA/D with differential-evolution operator
%                   (one of the most-cited, strongest MOEA/D variants in the
%                   literature -- added as a second "strong as real MOEA/D" peer)
%   CTAEA     2018  K. Li, R. Chen, G. Fu, X. Yao — two-archive EA for CMOPs
%   CCMO      2021  Y. Tian et al. — coevolutionary constrained MO framework
%   DCNSGAIII 2021  R. Jiao et al. — dynamic constrained NSGA-III
%                   (covers the encadrante's explicit NSGA-III request, natively)
%   BiCo      2022  Z. Liu, B. Wang, K. Tang — bidirectional coevolution
%   CMOEAMS   2022  Y. Tian et al. — multiple-stage constrained MOEA
%   MSCMO     2021  H. Ma et al. — multi-stage constrained MOEA
%
% Same fair protocol as every other external competitor already in the thesis:
%   20 runs x 9 instances, maxFE = 200000, pop = 100, rng(run) seeded,
%   feasible + non-dominated fronts only, saved as PROFITS (maximization).
%
% SETUP (MATLAB Online) — identical to the RL run that already worked:
%   addpath(genpath('/MATLAB Drive/PlatEMO-master'));
%   cd('/MATLAB Drive/comparison/competitors/platemo')
%   run_platemo_extra
% RESUMABLE: re-running skips finished (algorithm x instance) pairs and resumes
% a half-done one at the exact run where it stopped.

RLDIR = fileparts(mfilename('fullpath'));   % absolute folder of THIS script -- platemo()
                                             % changes the working directory internally, so
                                             % every path below is absolute, and we cd(RLDIR)
                                             % back before touching any file, every time.
NRUNS = 20;
MAXFE = 200000;
POP   = 100;

% ORDERED BY PRIORITY. Realistically you only need the first FEW:
%   moeadde = a second "strong, well-known" peer for real MOEA/D (2009, one of
%             the most-cited MOEA/D variants). NOTE: unlike the others below,
%             it is not tagged <constrained> in PlatEMO -- if it can't produce
%             feasible knapsack solutions on its own, the script's existing
%             feasibility check (see "no feasible solution, skipped" below)
%             will make that obvious rather than silently failing.
%   dcnsga3 = the only genuine GAP in the competitor set (a clean, native
%             NSGA-III -- the encadrante asked for NSGA-III explicitly, and the
%             pymoo one required an injected repair, so it is not compliant)
%   ctaea   = the most established constrained MOEA of the set (2018, IEEE TEVC)
% Everything below those three is OPTIONAL "nice to have" -- stop whenever you
% want, the script is resumable and each finished pair is already saved.
ALGS = { @MOEADDE,    'moeadde'   ;
         @DCNSGAIII, 'dcnsga3'   ;
         @CTAEA,     'ctaea'     ;
         @CCMO,      'ccmo'      ;
         @BiCo,      'bico'      ;
         @CMOEAMS,   'cmoeams'   ;
         @MSCMO,     'mscmo'     };

INST = { 2,250,'250.2' ; 3,250,'250.3' ; 4,250,'250.4' ;
         2,500,'500.2' ; 3,500,'500.3' ; 4,500,'500.4' ;
         2,750,'750.2' ; 3,750,'750.3' ; 4,750,'750.4' };

outdir = fullfile(RLDIR, 'extra_fronts');
if ~exist(outdir,'dir'); mkdir(outdir); end

for a = 1:size(ALGS,1)
    algfun = ALGS{a,1};  tag = ALGS{a,2};
    for k = 1:size(INST,1)
        M = INST{k,1};  D = INST{k,2};  name = INST{k,3};
        rawf = fullfile(outdir, sprintf('raw_%s_%s.txt',  tag, name));
        szf  = fullfile(outdir, sprintf('sizes_%s_%s.txt',tag, name));
        if exist(rawf,'file')
            fprintf('%s %s: done, skip\n', tag, name);  continue;
        end
        tmpraw = [rawf '.part'];  tmpsz = [szf '.part'];
        done = 0;                                  % resume: count finished runs
        if exist(tmpsz,'file')
            s = fileread(tmpsz);  done = sum(s == newline);
            fprintf('%s %s: resuming at run %d\n', tag, name, done+1);
        end
        cd(RLDIR);
        inst = load(sprintf('MOKPX-M%d-D%d.mat',M,D), 'P');
        psum = sum(inst.P,2)';
        for run = done+1:NRUNS
            rng(run,'twister');
            t0 = tic;
            try
                [~,Obj,Con] = platemo('algorithm',algfun,'problem',@MOKPX, ...
                    'M',M,'D',D,'N',POP,'maxFE',MAXFE);
            catch ME
                cd(RLDIR);
                fprintf(2,'%s %s run %d FAILED: %s\n', tag,name,run,ME.message);
                fprintf(2,'(skipping the REST of this algorithm and moving to the next one)\n');
                break
            end
            cd(RLDIR);                              % restore cwd BEFORE any file write
            feas = all(Con <= 1e-6, 2);
            if ~any(feas)
                fprintf(2,'%s %s run %d: no feasible solution, skipped\n', tag,name,run);
                continue
            end
            F = repmat(psum, sum(feas), 1) - Obj(feas,:);   % back to PROFITS (max)
            F = nd_max(F);
            fid = fopen(tmpraw,'a');
            fprintf(fid, [repmat('%.2f ',1,M) '\n'], F');
            fclose(fid);
            fid = fopen(tmpsz,'a');
            fprintf(fid, '%d\n', size(F,1));
            fclose(fid);
            fprintf('%s %s run %2d/%d  |F|=%3d  (%.1f min)\n', ...
                tag, name, run, NRUNS, size(F,1), toc(t0)/60);
        end
        if exist(tmpsz,'file') && sum(fileread(tmpsz)==newline) >= NRUNS
            movefile(tmpraw, rawf);  movefile(tmpsz, szf);
            fprintf('%s %s: COMPLETE -> %s\n', tag, name, rawf);
        end
    end
end
fprintf('\nALL DONE. Download the "%s" folder and send it back for scoring.\n', outdir);

%% ---- helpers ----------------------------------------------------------
function F = nd_max(F)
% unique rows, then keep the non-dominated subset (MAXIMIZATION)
F = unique(F,'rows');
n = size(F,1);  keep = true(n,1);
for i = 1:n
    for j = 1:n
        if j ~= i && keep(i) && all(F(j,:) >= F(i,:)) && any(F(j,:) > F(i,:))
            keep(i) = false;  break;
        end
    end
end
F = F(keep,:);
end
