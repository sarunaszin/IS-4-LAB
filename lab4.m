clc;
clear;
close all;

% Struktura:
%
% Mokymo vaizdas
%       |
%       v
% Simboliu segmentavimas
%       |
%       v
% 35 pozymiai
%       |
%       v
% Neuroninis tinklas
%       |
%       v
% 10 isejimu
%       |
%       v
% Atpazintas skaicius 0-9

%% MOKYMO VAIZDAS

% Mokymo paveiksle yra 5 eilutes skaiciu:
%
% 0 1 2 3 4 5 6 7 8 9
% 0 1 2 3 4 5 6 7 8 9
% 0 1 2 3 4 5 6 7 8 9
% 0 1 2 3 4 5 6 7 8 9
% 0 1 2 3 4 5 6 7 8 9

pavadinimas = 'train_data.png';

% Mokymo vaizde esanciu eiluciu skaicius

eiluciu_sk = 5;

%% POZYMIU ISSKYRIMAS MOKYMUI

% Funkcija:
%
% 1. Nuskaito vaizda
% 2. Paverčia ji dvejetainiu
% 3. Suranda atskirus simbolius
% 4. Iskerpa simbolius
% 5. Suvienodina ju dydi iki 70x50
% 6. Padalija i 7x5 dalis
% 7. Apskaiciuoja 35 pozymius

pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, eiluciu_sk);

%% POZYMIU MATRICA

% Pozymiai is celiu masyvo perkeliami i matrica.
%
% Kiekvienas stulpelis atitinka viena skaiciu.
%
% 5 eilutes * 10 skaiciu = 50 pavyzdziu
%
% P dydis turetu buti 35 x 50

P = cell2mat(pozymiai_tinklo_mokymui);

%% MOKYMO DUOMENU PATIKRA

disp(' ');
disp('MOKYMO DUOMENYS');
disp(' ');

fprintf('Aptikta simboliu: %d\n', size(P,2));
fprintf('Pozymiu vienam simboliui: %d\n', size(P,1));

tiketinas_simboliu_sk = eiluciu_sk * 10;

if size(P,2) ~= tiketinas_simboliu_sk
    warning('Tiketasi %d simboliu, bet aptikta %d.', tiketinas_simboliu_sk, size(P,2));
end

%% PAGEIDAUJAMU ATSAKYMU MATRICA

% Turime 10 klasiu:
%
% 0 1 2 3 4 5 6 7 8 9
%
% Kiekvienai klasei naudojamas vienas is 10 isejimu.
%
% Pvz.:
%
% 0 -> [1 0 0 0 0 0 0 0 0 0]'
% 1 -> [0 1 0 0 0 0 0 0 0 0]'
%
% Kadangi kiekvienoje eiluteje skaiciai surasyti
% nuo 0 iki 9, eye(10) kartojama 5 kartus.

T = repmat(eye(10), 1, eiluciu_sk);

fprintf('P dydis = %d x %d\n', size(P,1), size(P,2));
fprintf('T dydis = %d x %d\n', size(T,1), size(T,2));

%% RBF TINKLAS SU 13 NEURONU

% newrb(P,T,goal,spread,MN)
%
% P      - mokymo pozymiai
% T      - norimi atsakymai
% goal   - norima klaida
% spread - RBF funkciju plotis
% MN     - maksimalus neuronu skaicius

neuronu_sk_1 = 13;

tinklas_RBF_13 = newrb(P,T,0,1,neuronu_sk_1);

%% RBF TINKLAS SU MAZESNIU NEURONU SKAICIUMI

% Pagal uzduoti sumazinamas RBF neuronu skaicius.

neuronu_sk_2 = 8;

tinklas_RBF_8 = newrb(P,T,0,1,neuronu_sk_2);

%% MLP - PAPILDOMA UZDUOTIS

% Daugiasluoksnis perceptronas.
%
% Pasleptame sluoksnyje pasirenkama 10 neuronu.

pasleptu_neuronu_sk = 10;

rng(1);

tinklas_MLP = feedforwardnet(pasleptu_neuronu_sk);

% Kadangi tikrinimui turime atskira nuotrauka,
% visi mokymo pavyzdziai gali buti naudojami mokymui.

tinklas_MLP.divideFcn = 'dividetrain';

tinklas_MLP = train(tinklas_MLP,P,T);

%% MOKYMO DUOMENU PATIKRA

% Patikriname, kaip tinklai atpazista
% mokymui naudotus pavyzdzius.

Y_train_RBF_13 = sim(tinklas_RBF_13,P);
[~, klases_train_RBF_13] = max(Y_train_RBF_13);

Y_train_RBF_8 = sim(tinklas_RBF_8,P);
[~, klases_train_RBF_8] = max(Y_train_RBF_8);

Y_train_MLP = sim(tinklas_MLP,P);
[~, klases_train_MLP] = max(Y_train_MLP);

% Tikros mokymo klases

[~, tikros_train_klases] = max(T);

% Mokymo tikslumas

tikslumas_train_RBF_13 = 100 * sum(klases_train_RBF_13 == tikros_train_klases) / length(tikros_train_klases);

tikslumas_train_RBF_8 = 100 * sum(klases_train_RBF_8 == tikros_train_klases) / length(tikros_train_klases);

tikslumas_train_MLP = 100 * sum(klases_train_MLP == tikros_train_klases) / length(tikros_train_klases);

disp(' ');
disp('MOKYMO PAVYZDZIU ATPAZINIMAS');
disp(' ');

fprintf('RBF su %d neuronu: %.2f %%\n', neuronu_sk_1, tikslumas_train_RBF_13);
fprintf('RBF su %d neuronais: %.2f %%\n', neuronu_sk_2, tikslumas_train_RBF_8);
fprintf('MLP: %.2f %%\n', tikslumas_train_MLP);

%% TESTAVIMO VAIZDAS

% Naudojama atskira nuotrauka su mokymo metu
% nematytais ranka rasytais skaiciais.

pavadinimas_testui = 'test_telefonas.png';

%% TESTAVIMO POZYMIU ISSKYRIMAS

% Testavimo paveiksle yra viena skaiciu eilute.

pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas_testui,1);

P_test = cell2mat(pozymiai_patikrai);

disp(' ');

fprintf('Testavimo vaizde aptikta simboliu: %d\n', size(P_test,2));

%% RBF 13 TESTAVIMAS

Y_RBF_13 = sim(tinklas_RBF_13,P_test);

% Randame didziausia kiekvieno simbolio isejima.

[~, klases_RBF_13] = max(Y_RBF_13);

% MATLAB klases yra 1-10,
% o musu skaiciai yra 0-9.
%
% 1 klase -> 0
% 2 klase -> 1
% ...
% 10 klase -> 9

skaiciai_RBF_13 = klases_RBF_13 - 1;

atsakymas_RBF_13 = sprintf('%d',skaiciai_RBF_13);

%% RBF 8 TESTAVIMAS

Y_RBF_8 = sim(tinklas_RBF_8,P_test);

[~, klases_RBF_8] = max(Y_RBF_8);

skaiciai_RBF_8 = klases_RBF_8 - 1;

atsakymas_RBF_8 = sprintf('%d',skaiciai_RBF_8);

%% MLP TESTAVIMAS

Y_MLP = sim(tinklas_MLP,P_test);

[~, klases_MLP] = max(Y_MLP);

skaiciai_MLP = klases_MLP - 1;

atsakymas_MLP = sprintf('%d',skaiciai_MLP);

%% ATPAZINIMO REZULTATAI

disp(' ');
disp('ATPAZINIMO REZULTATAI');
disp(' ');

fprintf('RBF su %d neuronu: %s\n', neuronu_sk_1, atsakymas_RBF_13);
fprintf('RBF su %d neuronais: %s\n', neuronu_sk_2, atsakymas_RBF_8);
fprintf('MLP: %s\n', atsakymas_MLP);

%% REZULTATU ATVAIZDAVIMAS

figure;

text(0.05,0.75,['RBF 13: ' atsakymas_RBF_13],'FontSize',24);
text(0.05,0.50,['RBF 8: ' atsakymas_RBF_8],'FontSize',24);
text(0.05,0.25,['MLP: ' atsakymas_MLP],'FontSize',24);

axis off;

title('Ranka rasytu skaiciu atpazinimas');

%% TESTAVIMO TIKSLUMAS

% Cia galima irasyti tikra testavimo nuotraukoje
% esancia skaiciu seka.
%
% Reiksme rasoma kaip tekstas, kad nebutu
% prarastas pirmasis 0.
%
% Pvz.:
%
% teisingas_atsakymas = '0123456789';
%
% Jei tikslumo skaiciuoti nenorime:
%
% teisingas_atsakymas = '';

teisingas_atsakymas = '047215639';

if ~isempty(teisingas_atsakymas)

    if length(teisingas_atsakymas) ~= size(P_test,2)

        warning('Teisingame atsakyme yra %d simboliu, bet testavimo vaizde aptikta %d.', length(teisingas_atsakymas), size(P_test,2));

    else

        % RBF 13 tikslumas

        teisingi_RBF_13 = sum(atsakymas_RBF_13 == teisingas_atsakymas);

        tikslumas_RBF_13 = 100 * teisingi_RBF_13 / length(teisingas_atsakymas);


        % RBF 8 tikslumas

        teisingi_RBF_8 = sum(atsakymas_RBF_8 == teisingas_atsakymas);

        tikslumas_RBF_8 = 100 * teisingi_RBF_8 / length(teisingas_atsakymas);


        % MLP tikslumas

        teisingi_MLP = sum(atsakymas_MLP == teisingas_atsakymas);

        tikslumas_MLP = 100 * teisingi_MLP / length(teisingas_atsakymas);


        % Rezultatai

        disp(' ');
        disp('TESTAVIMO TIKSLUMAS');
        disp(' ');

        fprintf('Tikras atsakymas: %s\n', teisingas_atsakymas);
        fprintf('RBF su %d neuronu: %.2f %%\n', neuronu_sk_1, tikslumas_RBF_13);
        fprintf('RBF su %d neuronais: %.2f %%\n', neuronu_sk_2, tikslumas_RBF_8);
        fprintf('MLP: %.2f %%\n', tikslumas_MLP);


        % Tikslumo grafikas

        figure;

        bar([tikslumas_RBF_13 tikslumas_RBF_8 tikslumas_MLP]);

        set(gca,'XTickLabel',{'RBF 13','RBF 8','MLP'});

        ylim([0 100]);

        ylabel('Tikslumas, %');

        title('Neuroniniu tinklu atpazinimo tikslumas');

        grid on;

    end

end