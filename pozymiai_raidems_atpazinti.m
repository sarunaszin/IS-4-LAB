function pozymiai = pozymiai_raidems_atpazinti(pavadinimas, pvz_eiluciu_sk)

%% Funkcijos taikymo pavyzdys

% pozymiai = pozymiai_raidems_atpazinti('test_data.png', 8);


%% Vaizdo su pavyzdžiais nuskaitymas

V = imread(pavadinimas);

figure(12), imshow(V)


%% Simbolių iškirpimas ir sudėjimas į kintamojo 'objektai' celes

% Spalvotas vaizdas pakeičiamas į pustonį

V_pustonis = rgb2gray(V);


% Apskaičiuojama slenkstinė reikšmė vaizdui pakeisti į dvejetainį

slenkstis = graythresh(V_pustonis);


% Pustonio vaizdas pakeičiamas į dvejetainį

V_dvejetainis = im2bw(V_pustonis,slenkstis);


% Rezultato atvaizdavimas

figure(1), imshow(V_dvejetainis)


% Vaizde esančių objektų kontūrų paieška

V_konturais = edge(uint8(V_dvejetainis));


% Rezultato atvaizdavimas

figure(2), imshow(V_konturais)


% Objektų kontūrų užpildymas

se = strel('square',7); % struktūrinis elementas užpildymui

V_uzpildyti = imdilate(V_konturais,se);


% Rezultato atvaizdavimas

figure(3), imshow(V_uzpildyti)


% Tuštumų objektų viduje užpildymas

V_vientisi = imfill(V_uzpildyti,'holes');


% Rezultato atvaizdavimas

figure(4), imshow(V_vientisi)


% Vientisų objektų dvejetainiame vaizde numeravimas

[O_suzymeti Skaicius] = bwlabel(V_vientisi);


% Apskaičiuojami dvejetainiame vaizde esančių objektų požymiai

O_pozymiai = regionprops(O_suzymeti);


% Nuskaitomos objektų ribų koordinatės

O_ribos = [O_pozymiai.BoundingBox];


% Kadangi ribą nusako 4 koordinatės, reikšmės pergrupuojamos

O_ribos = reshape(O_ribos,[4 Skaicius]);


% Nuskaitomos objektų centro koordinatės

O_centras = [O_pozymiai.Centroid];


% Kadangi centrą nusako 2 koordinatės, reikšmės pergrupuojamos

O_centras = reshape(O_centras,[2 Skaicius]);

O_centras = O_centras';


% Kiekvienam objektui pridedamas numeris
% Trečiame stulpelyje saugomas objekto numeris

O_centras(:,3) = 1:Skaicius;


% Objektai surūšiuojami pagal vertikalią koordinatę

O_centras = sortrows(O_centras,2);


% Objektai rūšiuojami pagal pavyzdžių eilučių
% ir simbolių skaičių eilutėje

raidziu_sk = Skaicius/pvz_eiluciu_sk;

for k = 1:pvz_eiluciu_sk

    O_centras((k-1)*raidziu_sk+1:k*raidziu_sk,:) = ...
        sortrows(O_centras((k-1)*raidziu_sk+1:k*raidziu_sk,:),3);

end


%% Simbolių iškirpimas

% Iš dvejetainio vaizdo pagal objektų ribas
% iškerpami atskiri simboliai

for k = 1:Skaicius

    objektai{k} = imcrop(V_dvejetainis,O_ribos(:,O_centras(k,3)));

end


% Iškirptų simbolių atvaizdavimas

figure(5),

for k = 1:Skaicius

    subplot(pvz_eiluciu_sk,raidziu_sk,k), imshow(objektai{k})

end


%% Nereikalingo balto fono pašalinimas

% Vaizdo fragmentai apkerpami pašalinant baltą foną iš kraštų

for k = 1:Skaicius

    V_fragmentas = objektai{k];


    % Nustatomas kiekvieno vaizdo fragmento dydis

    [aukstis, plotis] = size(V_fragmentas);


    % Baltų stulpelių pašalinimas

    stulpeliu_sumos = sum(V_fragmentas,1);

    V_fragmentas(:,stulpeliu_sumos == aukstis) = [];


    % Perskaičiuojamas objekto dydis

    [aukstis, plotis] = size(V_fragmentas);


    % Baltų eilučių pašalinimas

    eiluciu_sumos = sum(V_fragmentas,2);

    V_fragmentas(eiluciu_sumos == plotis,:) = [];


    % Apkarpytas vaizdas išsaugomas vietoje senojo

    objektai{k} = V_fragmentas;

end


% Apkarpytų simbolių atvaizdavimas

figure(6),

for k = 1:Skaicius

    subplot(pvz_eiluciu_sk,raidziu_sk,k), imshow(objektai{k})

end


%% Vaizdo fragmentų dydžio suvienodinimas iki 70x50

for k = 1:Skaicius

    V_fragmentas = objektai{k];


    % Simbolio dydis pakeičiamas į 70x50

    V_fragmentas_7050 = imresize(V_fragmentas,[70,50]);


    % Vaizdo fragmentas padalijamas į 10x10 dydžio dalis

    for m = 1:7

        for n = 1:5


            % Apskaičiuojama kiekvienos 10x10 dalies
            % šviesumo reikšmių suma

            Vid_sviesumas_eilutese = ...
                sum(V_fragmentas_7050((m*10-9:m*10),(n*10-9:n*10)));


            Vid_sviesumas((m-1)*5+n) = ...
                sum(Vid_sviesumas_eilutese);

        end

    end


    % 10x10 dydžio dalyje didžiausia galima
    % šviesumo reikšmių suma yra 100
    %
    % Reikšmės normuojamos į intervalą [0,1]

    Vid_sviesumas = (100-Vid_sviesumas)/100;


    % Požymius neuronų tinklui patogiau pateikti stulpeliu

    Vid_sviesumas = Vid_sviesumas(:);


    % Apskaičiuoti požymiai išsaugomi į bendrą kintamąjį

    pozymiai{k} = Vid_sviesumas;

end
