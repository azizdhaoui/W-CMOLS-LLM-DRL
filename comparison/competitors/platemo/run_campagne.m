function run_campagne(nameStr)
% Campagne complete d'UN algorithme PlatEMO : 20 executions x 9 instances,
% 200000 evaluations, population egale a celle qu'utilise MOEA/D sur
% l'instance. Formulation MOKPR (reparation gloutonne native de PlatEMO).
%
% Chaque execution ecrit immediatement son front ET son temps, et un couple
% deja calcule est saute : la campagne est donc reprenable a tout moment.

PILOT = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(PILOT,'..','pl','PlatEMO-master','PlatEMO')));
addpath(PILOT);

MAXFE = 200000;
NRUNS = 20;
INST = { 2,250,'250.2',150 ; 3,250,'250.3',200 ; 4,250,'250.4',250 ;
         2,500,'500.2',200 ; 3,500,'500.3',250 ; 4,500,'500.4',300 ;
         2,750,'750.2',250 ; 3,750,'750.3',300 ; 4,750,'750.4',350 };

tag = lower(nameStr);
out = fullfile(PILOT,'campagnes',tag);
if ~exist(out,'dir'); mkdir(out); end

for k = 1:size(INST,1)
    M = INST{k,1}; D = INST{k,2}; iname = INST{k,3}; POP = INST{k,4};
    cd(PILOT);
    inst = load(sprintf('MOKPX-M%d-D%d.mat',M,D),'P');
    psum = sum(inst.P,2)';
    for run = 1:NRUNS
        ff = fullfile(out, sprintf('front_%s_r%d.txt', iname, run));
        tf = fullfile(out, sprintf('time_%s_r%d.txt',  iname, run));
        if exist(ff,'file'); continue; end
        rng(run,'twister');
        t0 = tic;
        try
            [~,Obj] = platemo('algorithm',str2func(nameStr),'problem',@MOKPR, ...
                'M',M,'D',D,'N',POP,'maxFE',MAXFE);
            cd(PILOT);
            secs = toc(t0);
            if isempty(Obj)
                fprintf(2,'%s %s run%d : aucune solution\n',tag,iname,run);
                continue
            end
            F = unique(repmat(psum,size(Obj,1),1) - Obj, 'rows');
            fid = fopen(ff,'w');
            fprintf(fid, [repmat('%.2f ',1,M) char(10)], F');
            fclose(fid);
            fid = fopen(tf,'w'); fprintf(fid,'%.3f%s',secs,char(10)); fclose(fid);
            fprintf('%s %s run %2d/%d  |F|=%-5d %6.1fs\n', tag, iname, run, NRUNS, size(F,1), secs);
        catch ME
            cd(PILOT);
            fprintf(2,'%s %s run%d ERREUR: %s\n',tag,iname,run,ME.message);
        end
    end
end
fprintf('CAMPAGNE %s TERMINEE\n', upper(tag));
end
