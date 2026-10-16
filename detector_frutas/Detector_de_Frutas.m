function Detector_de_Frutas()

    pkg load image

    % =====================================================
    % CONFIGURACAO: ALTERE AQUI A IMAGEM A SER ANALISADA
    % =====================================================

    arquivo_teste = 'teste.jpg';

    % =====================================================
    % 1. CARREGAMENTO DAS IMAGENS
    % =====================================================

    pasta = 'frutas';

    classes = {'banana', 'laranja', 'maca'};

    Xcor = [];
    Xforma = [];
    Xborda = [];
    y = [];
    arquivos = {};

    exts = {'*.jpg', '*.jpeg', '*.png', '*.webp', '*.avif', ...
         '*.JPG', '*.JPEG', '*.PNG', '*.WEBP', '*.AVIF'};

    for c = 1:numel(classes)

        lista = {};

        for e = 1:numel(exts)

            a = dir(fullfile(pasta, classes{c}, exts{e}));

            for j = 1:numel(a)

                lista{end+1} = fullfile( ...
                    pasta, classes{c}, a(j).name);

            end

        end

        lista = unique(lista);

        for j = 1:numel(lista)

            [cor, forma, borda] = ...
                extrair_caracteristicas(lista{j});

            Xcor = [Xcor; cor];
            Xforma = [Xforma; forma];
            Xborda = [Xborda; borda];

            y(end+1,1) = c;

            arquivos{end+1} = lista{j};

        end

    end

    if isempty(y)
        error('Nenhuma imagem encontrada na pasta frutas.');
    end

    printf('\nImagens carregadas: %d\n', numel(y));

    for c = 1:numel(classes)

        printf('%s: %d imagens\n', ...
            classes{c}, sum(y == c));

    end


    % =====================================================
    % 2. COMPARACAO DAS CARACTERISTICAS
    % =====================================================

    conjuntos = {
        Xcor,
        Xforma,
        [Xcor, Xforma],
        [Xcor, Xforma, Xborda]
    };

    nomes_conj = {
        'Apenas cor',
        'Apenas forma',
        'Cor + forma',
        'Cor + forma + bordas'
    };

    k = 3;

    acuracias = zeros(1, numel(conjuntos));

    printf('\nComparacao de caracteristicas:\n');

    for s = 1:numel(conjuntos)

        acuracias(s) = loo_acuracia( ...
            conjuntos{s}, y, k);

        printf('%-25s %.2f%%\n', ...
            nomes_conj{s}, acuracias(s));

    end


    % =====================================================
    % 3. ESCOLHA DO MELHOR CONJUNTO
    % =====================================================

    [~, melhor] = max(acuracias);

    X = conjuntos{melhor};

    printf('\nMelhor conjunto: %s\n', ...
        nomes_conj{melhor});


    % =====================================================
    % 4. AVALIACAO FINAL
    % =====================================================

    n = numel(y);

    predito = zeros(n,1);

    for i = 1:n

        treino = true(n,1);

        treino(i) = false;

        predito(i) = knn_predizer( ...
            X(treino,:), ...
            y(treino), ...
            X(i,:), ...
            k);

    end

    acuracia = mean(predito == y) * 100;

    printf('\nAcuracia geral: %.2f%%\n', acuracia);

    recalls = zeros(1, numel(classes));

    for c = 1:numel(classes)

        recalls(c) = mean(predito(y == c) == c) * 100;

        printf('%-10s %.2f%%\n', ...
            classes{c}, recalls(c));

    end

    printf('Acuracia balanceada: %.2f%%\n', ...
        mean(recalls));


    % =====================================================
    % 5. MATRIZ DE CONFUSAO
    % =====================================================

    nc = numel(classes);

    conf = accumarray([y predito], 1, [nc nc]);

    printf('\nMatriz de confusao:\n');

    disp(conf);

    printf('Linhas = classe real\n');
    printf('Colunas = classe prevista\n');


    % =====================================================
    % 6. CLASSIFICAR UMA NOVA IMAGEM
    % =====================================================

    if exist(arquivo_teste, 'file')

        % Aviso: imagem de teste nao pode estar no treino
        [~, nome_teste, ext_teste] = fileparts(arquivo_teste);

        for i = 1:numel(arquivos)

            [~, nome_i, ext_i] = fileparts(arquivos{i});

            if strcmp([nome_i ext_i], [nome_teste ext_teste])

                printf(['\nATENCAO: existe um arquivo com o mesmo ' ...
                    'nome no treino: %s\n'], arquivos{i});

            end

        end

        [cor, forma, borda] = ...
            extrair_caracteristicas(arquivo_teste);

        % Mostrar caracteristicas extraidas
        printf('\nCaracteristicas da imagem de teste:\n');

        printf('Cor:\n');
        disp(cor);

        printf('Forma:\n');
        disp(forma);

        printf('Bordas:\n');
        disp(borda);

        % Selecionar caracteristicas
        if melhor == 1

            x = cor;

        elseif melhor == 2

            x = forma;

        elseif melhor == 3

            x = [cor, forma];

        else

            x = [cor, forma, borda];

        end

        % Classificar imagem
        [classe, d, ord] = knn_predizer(X, y, x, k);

        printf('\nFruta identificada: %s\n', ...
            classes{classe});

        % Diagnostico: vizinhos mais proximos
        printf('\nVizinhos mais proximos:\n');

        for i = 1:min(5, numel(ord))

            printf('%s  (classe: %s, dist: %.3f)\n', ...
                arquivos{ord(i)}, ...
                classes{y(ord(i))}, ...
                d(ord(i)));

        end

        % Mostrar imagem e mascara (MESMO arquivo classificado)
        [img, mascara] = preprocessar(arquivo_teste);

        figure;

        subplot(1,2,1);
        imshow(img);
        title(['Fruta identificada: ' classes{classe}]);

        subplot(1,2,2);
        imshow(mascara);
        title('Mascara da fruta');

    else

        printf('\nArquivo "%s" nao encontrado. Verifique o nome em arquivo_teste no inicio do codigo.\n', arquivo_teste);

    end

end

% =====================================================
% 7. PRE-PROCESSAMENTO
% =====================================================

function [img, mascara] = preprocessar(arquivo)

    img = imread(arquivo);

    if size(img,3) == 1

        img = repmat(img, [1 1 3]);

    end

    img = img(:,:,1:3);

    % Normalizacao
    img = im2double(img);

    % Completa a imagem ate ficar quadrada (sem deformar a fruta)
    [h, w, ~] = size(img);
    dif = abs(h - w);
    p1 = floor(dif/2);
    p2 = dif - p1;

    if h > w
        img = padarray(img, [0 p1 0], 'replicate', 'pre');
        img = padarray(img, [0 p2 0], 'replicate', 'post');
    elseif w > h
        img = padarray(img, [p1 0 0], 'replicate', 'pre');
        img = padarray(img, [p2 0 0], 'replicate', 'post');
    end

    % Redimensionamento
    img = imresize(img, [200 200]);
    img = min(max(img, 0), 1);

    % Estimativa do fundo pelas bordas
    b = 8;

    borda = [
        reshape(img(1:b,:,:), [], 3);
        reshape(img(end-b+1:end,:,:), [], 3);
        reshape(img(:,1:b,:), [], 3);
        reshape(img(:,end-b+1:end,:), [], 3)
    ];

    fundo = median(borda, 1);

    dist = sqrt(sum( ...
        (img - reshape(fundo,1,1,3)).^2, 3));

    if max(dist(:)) == 0

        mascara = false(size(dist));

    else

        limiar = graythresh(dist / max(dist(:)));

        limiar = max(limiar, 0.12);

        mascara = dist > limiar;

        mascara = imfill(mascara, 'holes');

        mascara = bwareaopen(mascara, 200);

    end

    % Plano B: saturacao
    if ~mascara_valida(mascara)

        hsv = rgb2hsv(img);

        mascara = hsv(:,:,2) > 0.25;

        mascara = imfill(mascara, 'holes');

        mascara = bwareaopen(mascara, 200);

    end

    % Plano C: elipse central
    if ~mascara_valida(mascara)

        [r,c] = ndgrid(1:200, 1:200);

        mascara = ...
            ((r - 100) / 70).^2 + ...
            ((c - 100) / 70).^2 <= 1;

    end

    % Remove folhas verdes (so quando sao minoria da mascara,
    % para nao estragar macas verdes)
    hsv_m = rgb2hsv(img);

    verde = hsv_m(:,:,1) > 0.17 & hsv_m(:,:,1) < 0.50 & ...
            hsv_m(:,:,2) > 0.25;

    if any(mascara(:)) && mean(verde(mascara)) < 0.25

        m2 = mascara & ~verde;

        % Abertura para remover cabinhos e restos finos
        r = 5;
        [xx, yy] = meshgrid(-r:r);
        se = (xx.^2 + yy.^2) <= r^2;

        m2 = imopen(m2, se);
        m2 = imfill(m2, 'holes');

        if mascara_valida(m2)
            mascara = m2;
        end

    end

    % Mantem o maior objeto
    L = bwlabel(mascara);

    props = regionprops(L, 'Area');

    if isempty(props)

        mascara = true(200,200);

    else

        [~,idx] = max([props.Area]);

        mascara = L == idx;

    end

end


% =====================================================
% 8. VALIDACAO DA MASCARA
% =====================================================

function v = mascara_valida(m)

    L = bwlabel(m);

    if max(L(:)) == 0

        v = false;

        return;

    end

    areas = accumarray(L(L > 0), 1);

    cobertura = max(areas) / numel(m);

    v = cobertura >= 0.08 && cobertura <= 0.90;

end


% =====================================================
% 9. EXTRACAO DE CARACTERISTICAS
% =====================================================

function [cor, forma, bordas] = extrair_caracteristicas(arquivo)

    [img, mascara] = preprocessar(arquivo);

    hsv = rgb2hsv(img);

    Himg = hsv(:,:,1);
    Simg = hsv(:,:,2);
    Vimg = hsv(:,:,3);

    H = Himg(mascara);
    S = Simg(mascara);
    V = Vimg(mascara);

    % Caracteristicas de cor (mediana: robusta a folhas e sombras)
    cor = [
        median(sin(2*pi*H)), ...
        median(cos(2*pi*H)), ...
        median(S), ...
        median(V)
    ];

    % Caracteristicas de forma
    props = regionprops(mascara, 'Area', 'Perimeter', ...
        'MajorAxisLength', 'MinorAxisLength', 'Solidity');

    area = props(1).Area;

    perimetro = props(1).Perimeter;

    circularidade = ...
        4*pi*area / (perimetro^2 + eps);

    proporcao = props(1).MajorAxisLength / ...
        (props(1).MinorAxisLength + eps);

    solidez = props(1).Solidity;

    forma = [circularidade, proporcao, solidez];

    % Extracao de bordas com Sobel
    gray = rgb2gray(img);

    Gx = [
        -1  0  1;
        -2  0  2;
        -1  0  1
    ];

    Gy = [
        -1 -2 -1;
         0  0  0;
         1  2  1
    ];

    gx = conv2(gray, Gx, 'same');

    gy = conv2(gray, Gy, 'same');

    magnitude = sqrt(gx.^2 + gy.^2);

    limiar = 0.20 * max(magnitude(:));

    mapa_bordas = magnitude > limiar;

    % Proporcao de bordas na fruta
    bordas = mean(mapa_bordas(mascara));

end


% =====================================================
% 10. CLASSIFICADOR K-NN
% =====================================================

function [classe, d, ord] = knn_predizer(Xtr, ytr, x, k)

    % Normalizacao baseada no treinamento
    mu = mean(Xtr);

    sigma = std(Xtr);

    sigma(sigma == 0) = 1;

    Xn = (Xtr - mu) ./ sigma;

    xn = (x - mu) ./ sigma;

    % Distancia euclidiana
    d = sqrt(sum((Xn - xn).^2, 2));

    [~,ord] = sort(d);

    kk = min(k, numel(ord));

    viz = ytr(ord(1:kk));

    % Votacao ponderada
    pesos = 1 ./ (d(ord(1:kk)) + 1e-6);

    votos = accumarray(viz(:), pesos(:), ...
        [max(ytr), 1]);

    [~,classe] = max(votos);

end


% =====================================================
% 11. AVALIACAO LEAVE-ONE-OUT
% =====================================================

function acc = loo_acuracia(X, y, k)

    n = numel(y);

    pred = zeros(n,1);

    for i = 1:n

        treino = true(n,1);

        treino(i) = false;

        pred(i) = knn_predizer( ...
            X(treino,:), ...
            y(treino), ...
            X(i,:), ...
            k);

    end

    % Acuracia balanceada
    classes = unique(y);

    rec = zeros(numel(classes),1);

    for c = 1:numel(classes)

        rec(c) = mean( ...
            pred(y == classes(c)) == classes(c));

    end

    acc = mean(rec) * 100;

end
