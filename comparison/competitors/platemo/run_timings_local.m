function run_timings_local(platemoDir, npasses)
% Chronometre DCNSGA-III, CMODRL et MOEA-D-2WA sur les neuf instances, sur
% cette machine, avec le protocole de leurs campagnes de fronts : MOKPX
% (contraintes explicites), N = 100, 200 000 evaluations.
%
% Ces trois concurrents n'ont aucun temps complet : leurs fronts ont ete
% calcules par des scripts qui affichaient le temps sans l'ecrire.
%
% PAR PASSES. Une passe execute une fois chacun des 27 couples (algorithme,
% instance). La passe p utilise la graine p. On peut donc arreter apres
% n'importe quelle passe : chaque couple a alors le meme nombre de mesures.
% Le script est reprenable : un couple deja mesure pour la passe courante
% est saute.
%
% Rien d'autre ne doit tourner sur la machine pendant la mesure.
%
% usage : matlab -batch "run_timings_local('<dossier PlatEMO>', 20)"
% sortie : timings_local/times_<algo>_<instance>.txt, une ligne par passe.

RLDIR = fileparts(mfilename('fullpath'));
addpath(genpath(platemoDir));
addpath(RLDIR);

MAXFE = 200000;
POP   = 100;
ALGS  = { @DCNSGAIII, 'dcnsga3' ; @CMODRL, 'cmodrl' ; @MOEAD2WA, 'moead2wa' };
INST  = { 2,250,'250.2' ; 3,250,'250.3' ; 4,250,'250.4' ;
          2,500,'500.2' ; 3,500,'500.3' ; 4,500,'500.4' ;
          2,750,'750.2' ; 3,750,'750.3' ; 4,750,'750.4' };

out = fullfile(RLDIR, 'timings_local');
if ~exist(out, 'dir'); mkdir(out); end

% Echauffement non chronometre : chargement des classes et compilation JIT.
for a = 1:size(ALGS,1)
    cd(RLDIR);
    rng(0, 'twister');
    [~,~,~] = platemo('algorithm',ALGS{a,1},'problem',@MOKPX,'M',2,'D',250,'N',POP,'maxFE',2000);
    fprintf('%s echauffement fait\n', ALGS{a,2});
end

t_start = tic;
for pass = 1:npasses
    for a = 1:size(ALGS,1)
        for k = 1:size(INST,1)
            M = INST{k,1}; D = INST{k,2}; name = INST{k,3};
            tf = fullfile(out, sprintf('times_%s_%s.txt', ALGS{a,2}, name));
            done = 0;
            if exist(tf, 'file'); done = numel(load(tf)); end
            if done >= pass; continue; end
            cd(RLDIR);
            rng(pass, 'twister');
            t0 = tic;
            [~,~,~] = platemo('algorithm',ALGS{a,1},'problem',@MOKPX,'M',M,'D',D,'N',POP,'maxFE',MAXFE);
            secs = toc(t0);
            cd(RLDIR);
            fid = fopen(tf, 'a'); fprintf(fid, '%.3f\n', secs); fclose(fid);
            fprintf('%s passe %2d/%d %-9s %-6s %7.1f s | ecoule %5.1f min\n', ...
                    datestr(now,'HH:MM:SS'), pass, npasses, ALGS{a,2}, name, secs, toc(t_start)/60);
        end
    end
    fprintf('=== PASSE %d TERMINEE (%.1f min) ===\n', pass, toc(t_start)/60);
end
fprintf('CHRONOMETRAGE TERMINE\n');
end
