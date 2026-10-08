% run_rl_mokp.m — RL-combined MOEA competitors on the thesis's 9 exact
% Zitzler-Thiele MOKP instances, via PlatEMO. RESUMABLE: re-running the
% script skips everything already finished (per algorithm x instance, and
% resumes half-done ones at the exact run where it stopped).
%
% Algorithms (consumed directly from PlatEMO, no reimplementation):
%   DRLOSEMCMO — Ming et al. 2024, deep-RL(Q-learning)-assisted operator
%                selection for constrained MOEA (toolbox-light, runs first)
%   CMODRL     — 2025, constrained MO optimization via deep RL
%                (needs Deep Learning Toolbox; skipped with a warning if absent)
%
% Fair protocol (same as the thesis's 7-MOEA campaign):
%   20 runs x 9 instances, maxFE = 200000, pop = 100, rng(run) seeded,
%   feasible + non-dominated fronts only, saved as PROFITS (maximization)
%   in the thesis pipeline format: raw_<tag>_<inst>.txt + sizes_<tag>_<inst>.txt
%
% SETUP (MATLAB Online):
%   1) unzip('PlatEMO-master.zip');  addpath(genpath('PlatEMO-master'));
%   2) cd into the folder containing THIS file (and the MOKPX-*.mat files)
%   3) run_rl_mokp
% Progress prints per run with elapsed minutes. Stop anytime; re-run resumes.

RLDIR = fileparts(mfilename('fullpath'));   % absolute folder of THIS script -- platemo()
                                             % changes the working directory internally, so
                                             % every path below is absolute, and we cd(RLDIR)
                                             % back before touching any file, every time.
NRUNS = 20;
MAXFE = 200000;
POP   = 100;

ALGS = { @DRLOSEMCMO, 'drlosemcmo' ;
         @CMODRL,     'cmodrl'     };

% All 9 instances (extended 2026-07-18: the 2-obj-only run took ~3.5h in
% practice, far under the original ~25-30h estimate, so the full 9-instance
% comparison -- matching every other external competitor in the thesis -- is
% affordable). Already-finished 250.2/500.2/750.2 pairs are skipped below.
INST = { 2,250,'250.2' ; 3,250,'250.3' ; 4,250,'250.4' ;
         2,500,'500.2' ; 3,500,'500.3' ; 4,500,'500.4' ;
         2,750,'750.2' ; 3,750,'750.3' ; 4,750,'750.4' };

outdir = fullfile(RLDIR, 'rl_fronts');
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
                fprintf(2,'(if this mentions a missing toolbox/function, this algorithm\n');
                fprintf(2,' is unavailable on your MATLAB license — move on to the next.)\n');
                error('stopping so the message above is visible');
            end
            cd(RLDIR);                              % restore cwd BEFORE any file write
            feas = all(Con <= 1e-6, 2);
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
        movefile(tmpraw, rawf);  movefile(tmpsz, szf);
        fprintf('%s %s: COMPLETE -> %s\n', tag, name, rawf);
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
