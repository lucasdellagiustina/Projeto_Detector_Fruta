% Detector_de_Frutas.m
% Classificador simples de imagem de fruta: pré-processamento + k-NN (modelo leve)
% Estrutura de pastas esperada (na pasta atual ou dentro de "frutas/"):
%   banana/*.jpg   laranja/*.jpg   maca/*.jpg
% Para classificar uma foto nova, coloque "teste.jpg" na mesma pasta.

pkg load image

% =====================================================
% 1) PRÉ-PROCESSAMENTO
% =====================================================
% Segmentação principal: estima a cor do fundo pelas bordas da imagem
% e marca como fruta tudo que for suficientemente diferente dela.
function m = segmentar_por_fundo(img)
  d = double(img);
  b = 8;                                    % largura da borda (pixels)
  borda = [reshape(d(1:b,:,:), [], 3); ...
           reshape(d(end-b+1:end,:,:), [], 3); ...
           reshape(d(:,1:b,:), [], 3); ...
           reshape(d(:,end-b+1:end,:), [], 3)];
  fundo = median(borda, 1);                 % cor típica do fundo
  dist = sqrt(sum((d - reshape(fundo, 1, 1, 3)).^2, 3));

  if max(dist(:)) == 0
    m = false(size(dist));
    return;
  end

  % limiar: Otsu, mas nunca abaixo da variação natural do fundo
  limiar = graythresh(dist / max(dist(:))) * max(dist(:));
  limiar = max([limiar, 30, 2.5 * norm(std(borda, 0, 1))]);

  m = dist > limiar;
  m = imfill(m, 'holes');
  m = bwareaopen(m, 200);
end

% Plano B: método antigo, baseado em saturação
function m = segmentar_por_saturacao(img)
  hsv = rgb2hsv(img);
  m = hsv(:,:,2) > 0.25;
  m = imfill(m, 'holes');
  m = bwareaopen(m, 200);
end

% Uma máscara é plausível se o MAIOR objeto cobre entre 8% e 90% da imagem
function v = mascara_valida(m)
  L = bwlabel(m);
  if max(L(:)) == 0
    v = false;
    return;
  end
  area_maior = max(accumarray(L(L > 0), 1));   % tamanho do maior objeto
  c = area_maior / numel(m);
  v = (c >= 0.08) && (c <= 0.90);
end

% metodo: 1 = cor do fundo, 2 = saturação, 3 = elipse central (último recurso)
function [img, mascara, metodo] = preprocessar(arquivo)
  img = imread(arquivo);
  if size(img,3) == 1                 % imagem em tons de cinza
    img = repmat(img, [1 1 3]);
  end
  img = img(:,:,1:3);                 % descarta canal alfa (PNG)
  img = imresize(img, [200 200]);     % padroniza o tamanho

  metodo = 1;
  mascara = segmentar_por_fundo(img);
  if ~mascara_valida(mascara)                 % falhou: tenta o plano B
    mascara = segmentar_por_saturacao(img);
    metodo = 2;
  end
  if ~mascara_valida(mascara)                 % falhou de novo: plano C
    [r, c] = ndgrid(1:size(img,1), 1:size(img,2));
    mascara = ((r - size(img,1)/2) / (0.35*size(img,1))).^2 + ...
              ((c - size(img,2)/2) / (0.35*size(img,2))).^2 <= 1;
    metodo = 3;
  end

  if ~any(mascara(:))                 % segmentação falhou: usa a imagem toda
    warning('Segmentação falhou em %s; usando imagem inteira.', arquivo);
    mascara = true(200, 200);
  end

  % mantém apenas o maior objeto (a fruta)
  L = bwlabel(mascara);
  props = regionprops(L, 'Area');
  [~, idx] = max([props.Area]);
  mascara = (L == idx);
end

% =====================================================
% 2) EXTRAÇÃO DE CARACTERÍSTICAS (6 números por imagem)
% =====================================================
function feat = extrair_caracteristicas(arquivo)
  [img, mascara] = preprocessar(arquivo);
  hsv = rgb2hsv(img);
  H = hsv(:,:,1)(mascara);
  S = hsv(:,:,2)(mascara);
  V = hsv(:,:,3)(mascara);

  props = regionprops(mascara, 'Area', 'Perimeter', 'BoundingBox');
  bb = props(1).BoundingBox;
  circularidade = 4*pi*props(1).Area / (props(1).Perimeter^2);
  proporcao = bb(3) / bb(4);

  % matiz é circular: usa seno/cosseno para não confundir vermelho (0 e 1)
  feat = [mean(sin(2*pi*H)), mean(cos(2*pi*H)), mean(S), mean(V), ...
          circularidade, proporcao];
end

% =====================================================
% 3) MODELO: k-NN
% =====================================================
function classe = knn_predizer(Xtr, ytr, x, k)
  mu = mean(Xtr);
  sigma = std(Xtr);
  sigma(sigma == 0) = 1;               % evita divisão por zero
  Xn = (Xtr - mu) ./ sigma;
  xn = (x - mu) ./ sigma;

  d = sqrt(sum((Xn - xn).^2, 2));
  [~, ord] = sort(d);
  kk = min(k, numel(ord));
  viz = ytr(ord(1:kk));
  pesos = 1 ./ (d(ord(1:kk)) + 1e-6);      % vizinho mais próximo pesa mais
  votos = accumarray(viz(:), pesos(:), [max(ytr) 1]);
  [~, classe] = max(votos);
end

% =====================================================
% 4) CARREGAR AS IMAGENS DE TREINO
% =====================================================
if exist('frutas', 'dir')
  pasta = 'frutas';
else
  pasta = '.';
end

pastas = dir(pasta);
pastas = pastas([pastas.isdir] & ~ismember({pastas.name}, {'.','..'}));

X = []; y = []; nomes = {}; arquivos = {};
exts = {'*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG'};

for i = 1:numel(pastas)
  lista = {};
  for e = 1:numel(exts)
    a = dir(fullfile(pasta, pastas(i).name, exts{e}));
    for j = 1:numel(a)
      lista{end+1} = fullfile(pasta, pastas(i).name, a(j).name);
    end
  end
  lista = unique(lista);             % remove duplicatas (Windows ignora maiúsculas)
  if isempty(lista), continue; end   % ignora pastas sem imagens

  nomes{end+1} = pastas(i).name;
  c = numel(nomes);
  for j = 1:numel(lista)
    X = [X; extrair_caracteristicas(lista{j})];
    y(end+1, 1) = c;
    arquivos{end+1} = lista{j};
  end
end

if isempty(X)
  error('Nenhuma imagem encontrada em "%s" (pasta atual: %s).', pasta, pwd);
end

% avisa sobre imagens quase idênticas (inflam a avaliação)
Xz = (X - mean(X)) ./ (std(X) + eps);
for i = 1:size(X,1)-1
  for j = i+1:size(X,1)
    if norm(Xz(i,:) - Xz(j,:)) < 0.15
      printf('POSSÍVEL DUPLICATA: %s  e  %s\n', arquivos{i}, arquivos{j});
    end
  end
end

printf('\n%d imagens carregadas em %d classes:\n', size(X,1), numel(nomes));
for c = 1:numel(nomes)
  printf('  %-10s %d imagens\n', nomes{c}, sum(y == c));
end

% =====================================================
% 4b) DIAGNÓSTICO: máscara de cada imagem de treino
%     (máscara quase toda preta ou toda branca = segmentação falhou)
% =====================================================
figure('Name', 'Diagnóstico das máscaras');
nimg = numel(arquivos);
ncol = 6;
nlin = ceil(nimg / ncol);
printf('\nCobertura da máscara (fração da imagem que virou "fruta"):\n');
for i = 1:nimg
  [~, m, metodo] = preprocessar(arquivos{i});
  cobertura = mean(m(:));
  [~, nome_arq] = fileparts(arquivos{i});
  if metodo == 3 || cobertura < 0.05 || cobertura > 0.90
    extra = '';
    if metodo == 3, extra = '  <- segmentação falhou, usei o centro'; end
    printf('  ATENÇÃO %s  %.0f%%%s\n', arquivos{i}, cobertura*100, extra);
  end
  subplot(nlin, ncol, i);
  imshow(m);
  title(nome_arq, 'fontsize', 7, 'interpreter', 'none');
end
drawnow;

% =====================================================
% 5) AVALIAÇÃO: leave-one-out (ideal para poucas fotos)
% =====================================================
k = 3;
n = size(X, 1);

% Acurácia leave-one-out para um conjunto de características
function acc = loo_acuracia(X, y, k)
  n = size(X, 1);
  pred = zeros(n, 1);
  for i = 1:n
    treino = true(n, 1);
    treino(i) = false;
    pred(i) = knn_predizer(X(treino,:), y(treino), X(i,:), k);
  end
  % acurácia balanceada: média do acerto de cada classe
  % (não deixa a classe com mais fotos esconder o erro nas outras)
  rec = zeros(1, max(y));
  for c = 1:max(y)
    rec(c) = mean(pred(y == c) == c);
  end
  acc = mean(rec) * 100;
end

% Compara quais características ajudam mais
% colunas de X: 1=matiz(sen) 2=matiz(cos) 3=saturação 4=brilho 5=circularidade 6=proporção
conjuntos = {[1 2 3 4 5 6], [1 2 3 4], [1 2 3 4 5], [1 2 3], [1 2]};
nomes_conj = {'todas (cor + forma)', 'só cor (matiz, sat., brilho)', ...
              'cor + circularidade', 'matiz + saturação', 'só matiz'};
printf('\nComparação de características (acurácia balanceada, leave-one-out, k=%d):\n', k);
accs = zeros(1, numel(conjuntos));
for s = 1:numel(conjuntos)
  accs(s) = loo_acuracia(X(:, conjuntos{s}), y, k);
  printf('  %-30s %5.1f%%\n', nomes_conj{s}, accs(s));
end
[~, melhor] = max(accs);
cols = conjuntos{melhor};
printf('Usando: %s\n', nomes_conj{melhor});
X = X(:, cols);   % a partir daqui só as colunas escolhidas
predito = zeros(n, 1);

for i = 1:n
  treino = true(n, 1);
  treino(i) = false;                  % deixa a imagem i de fora
  predito(i) = knn_predizer(X(treino,:), y(treino), X(i,:), k);
end

acuracia = mean(predito == y) * 100;
printf('\nAcurácia geral (leave-one-out, k=%d): %.1f%%\n', k, acuracia);
referencia = max(accumarray(y, 1)) / numel(y) * 100;
printf('Referência (sempre chutar a classe maior): %.1f%%\n', referencia);
printf('Acerto por classe:\n');
recalls = zeros(1, numel(nomes));
for c = 1:numel(nomes)
  recalls(c) = mean(predito(y == c) == c) * 100;
  printf('  %-10s %5.1f%%  (%d de %d)\n', nomes{c}, recalls(c), ...
         sum(predito(y == c) == c), sum(y == c));
end
printf('Acurácia balanceada: %.1f%%\n', mean(recalls));

nc = numel(nomes);
conf = accumarray([y predito], 1, [nc nc]);
printf('\nMatriz de confusão (linha = real, coluna = predito):\n');
printf('%12s', ''); printf('%10s', nomes{:}); printf('\n');
for c = 1:nc
  printf('%-12s', nomes{c});
  printf('%10d', conf(c,:));
  printf('\n');
end

erros = find(predito ~= y);
if ~isempty(erros)
  printf('\nImagens classificadas errado:\n');
  for i = erros'
    printf('  %s -> previsto: %s\n', arquivos{i}, nomes{predito(i)});
  end
end

% =====================================================
% 6) CLASSIFICAR UMA FOTO NOVA (teste.jpg)
% =====================================================
if exist('teste.jpg', 'file')
  [img, mascara, metodo] = preprocessar('teste.jpg');
  x = extrair_caracteristicas('teste.jpg')(cols);
  c = knn_predizer(X, y, x, k);
  printf('\nFruta detectada em teste.jpg: %s\n', nomes{c});
  if metodo == 3
    printf('AVISO: não consegui separar a fruta do fundo; usei o centro da imagem.\n');
    printf('       Resultado pouco confiável. Prefira uma foto com fundo liso.\n');
  end

  figure;
  subplot(1, 2, 1); imshow(img);     title(['Detectada: ' nomes{c}]);
  subplot(1, 2, 2); imshow(mascara); title('Máscara (pré-processamento)');
else
  printf('\n(teste.jpg não encontrado em %s; pulei a classificação.)\n', pwd);
end
