import tkinter as tk
from tkinter import filedialog, messagebox
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Aqui fazemos o download do punkt,
# o qual é um tokenizer de texto, que é um modelo pré treinado para dividir um texto em sentenças ou palavras
nltk.download('punkt')

# Aqui carregamos o modelo chamado all-MiniLM-L6-v2 ,
# o qual é um modelo de transformer para gerar embeddings de sentenças
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Aqui temos a lista de possíveis temas para classificação de assuntos do texto
temas = [
    "família", "romance", "suspense", "educação", "história",
    "aventura", "mistério", "thriller", "terror", "policial",
    "drama", "biografia", "comédia", "fantasia", "ficção científica",
    "autoajuda", "ensino", "ensaios", "artigos", "relato",
    "jornalismo", "documentário", "teatro"
]

# Dicionário de palavras-chave para cada tema
palavras_chave = {
    "família": ["pais", "filhos", "amor", "casa", "relacionamento", "família", "irmãos", "cuidado", "educação",
                "tradição",
                "história", "lazer", "unidade", "apoio", "comunidade", "memórias", "valores", "herança", "cuidado",
                "solidão",
                "compreensão", "alegria", "festa", "nascimento", "perda"],

    "romance": ["amor", "paixão", "coração", "sentimento", "beijo", "relacionamento", "encontro", "desafio",
                "destino", "história", "complicado", "desejo", "traição", "cumplicidade", "conflito", "suspense",
                "felicidade", "encontros", "drama", "vínculo", "separação", "reencontro", "casamento", "noivado",
                "trauma"],

    "suspense": ["mistério", "tensão", "surpresa", "perigo", "investigação", "enigma", "sangue", "esconderijo",
                 "criminoso", "atmosfera", "segredo", "ação", "conspiração", "desespero", "dúvida", "ansiedade",
                 "trama", "desfecho", "perplexidade", "obstáculo", "resolução", "sinal", "sinalização", "desconfiança",
                 "reviravolta"],

    "educação": ["ensinar", "aprender", "professor", "aluno", "sala de aula", "matéria", "escola", "universidade",
                 "ensino", "conhecimento", "cultura", "habilidade", "formação", "desenvolvimento", "currículo",
                 "metodologia", "aprendizagem", "projeto", "aula", "livro", "literatura", "prática", "teoria", "tarefa",
                 "exame"],

    "história": ["passado", "evento", "cultura", "data", "memória", "narrativa", "perspectiva", "evolução",
                 "crônica", "época", "sociedade", "documento", "relato", "testemunho", "contexto", "registros",
                 "fatos", "personagem", "sistema", "conflito", "descoberta", "revolução", "reforma", "transição",
                 "legado"],

    "aventura": ["exploração", "perigo", "desafio", "descoberta", "jornada", "viagem", "experiência",
                 "ação", "deslocamento", "natureza", "excursão", "fuga", "trilha", "experimento", "competição",
                 "combate", "superação", "estratégia", "encontro", "suspense", "escapada", "caminho", "perseguição",
                 "tática", "destino"],

    "mistério": ["enigma", "segredo", "desconhecido", "suspense", "investigação", "indícios", "pista",
                 "tensão", "oculto", "misterioso", "inexplicável", "sombra", "interrogação", "encontrar",
                 "pergunta", "inquietude", "atmosfera", "abstrato", "conjectura", "solução", "complicação", "dilema",
                 "fantasia", "realidade"],

    "thriller": ["ação", "perigo", "tensão", "suspense", "emergência", "insegurança", "confusão",
                 "espectacular", "preocupação", "conflito", "clímax", "interação", "estratégia",
                 "jogo", "estrondo", "dilema", "reviravolta", "narrativa", "agitação", "suspense psicológico", "drama",
                 "conflito", "luta", "destino"],

    "terror": ["medo", "horror", "sangue", "fantasma", "maldição", "suspense", "sombra", "grito",
               "desespero", "pavor", "monstro", "pânico", "aflição", "sofrimento", "pesadelo",
               "assombração", "instinto", "paranoia", "tensão", "morte", "assassinato", "perigo", "escuridão", "agonia",
               "aflição"],

    "policial": ["crime", "investigação", "detetive", "suspeito", "cena do crime", "evidência", "conspiração",
                 "justiça", "mundo criminal", "interrogatório", "testemunha", "solução", "processo",
                 "método", "motivo", "preparação", "técnica", "disfarce", "perseguição", "captura",
                 "prisão", "prova", "condenação", "defesa", "julgamento"],

    "drama": ["emoção", "conflito", "relacionamento", "suspense", "experiência", "tensão", "decisão",
              "conflito interno", "vulnerabilidade", "dramático", "história de vida", "interação",
              "revelação", "crise", "superação", "comunicação", "vulnerabilidade", "alívio", "risada",
              "desapontamento", "mudança", "reflexão", "exploração", "autoanálise", "compreensão"],

    "biografia": ["vida", "história", "experiência", "personagem", "realidade", "trabalho",
                  "caminho", "sucesso", "luta", "legado", "influência", "desafios", "triunfo",
                  "decisões", "mudanças", "metas", "motivação", "inspirador", "narrativa",
                  "perspectiva", "cultura", "identidade", "autoestima", "crescimento", "destino"],

    "comédia": ["humor", "risada", "diversão", "ironia", "satira", "paródia", "brincadeira",
                "comédia romântica", "esquete", "pantomima", "ridículo", "engraçado",
                "alegria", "sátira", "diversão", "palhaçada", "festa", "alegria", "piada",
                "desastre", "engano", "quiproquó", "cômico", "bobagem", "trote"],

    "fantasia": ["mágico", "imaginário", "aventura", "ficção", "criatividade", "sonho",
                 "mundo mágico", "realidade alternativa", "seres fantásticos", "exploração",
                 "mitologia", "lenda", "encontro", "destino", "herói", "aventura",
                 "conflito", "ação", "mistério", "invenção", "conhecimento", "segredo", "paradoxo", "suspense", "fuga"],

    "ficção científica": ["futuro", "tecnologia", "inovação", "espaço", "alienígena", "realidade",
                          "exploração", "computação", "evolução", "experiência", "experimento",
                          "paradoxo", "mundo alternativo", "avançado", "hipótese", "teoria",
                          "distopia", "utopia", "viagem no tempo", "inteligência", "conflito", "existência",
                          "identidade", "alteridade"],

    "autoajuda": ["motivação", "crescimento", "autoconhecimento", "superação", "mudança",
                  "estratégia", "metas", "conselho", "inspirador", "desafios", "história",
                  "vida", "experiência", "reflexão", "foco", "resiliência",
                  "autoestima", "desenvolvimento", "sucesso", "ajuda", "orientação",
                  "perspectiva", "técnica", "teoria", "prática"],

    "ensino": ["educação", "aprender", "professor", "aluno", "metodologia",
               "habilidade", "tarefa", "material", "currículo", "escola",
               "universidade", "formação", "ensino", "literatura", "teoria",
               "prática", "habilidade", "desempenho", "exame", "programa", "ensaios",
               "artigos", "relato", "jornalismo", "documentário", "teatro"],

    "ensaios": ["análise", "discussão", "teoria", "opinião", "argumento",
                "evidência", "análise crítica", "estudo", "interpretação",
                "ponto de vista", "conclusão", "reflexão", "perspectiva",
                "argumentação", "julgamento", "justificativa", "defesa",
                "abordagem", "análise", "exposição", "tópico", "temática"],

    "artigos": ["pesquisa", "estudo", "informação", "discussão", "relatório",
                "tópico", "análise", "dados", "tendência", "artigo científico",
                "publicação", "revisão", "resultado", "conclusão", "análise crítica",
                "julgamento", "justificativa", "defesa", "abordagem",
                "temática", "instrução", "apresentação"],

    "relato": ["experiência", "vivência", "experimento", "relato pessoal",
               "experiência", "testemunho", "narrativa", "análise", "desafio",
               "história", "encontro", "testemunha", "perspectiva",
               "opinião", "reflexão", "interpretação", "ponto de vista",
               "conclusão", "resumo", "avaliação", "discussão"],

    "jornalismo": ["notícias", "reportagem", "entrevista", "fato", "análise",
                   "investigação", "editorial", "opinião", "jornal",
                   "reporte", "cobertura", "contexto", "narrativa",
                   "fatos", "destaque", "reportagem", "ponto de vista",
                   "reportagem investigativa", "análise crítica", "informação"],

    "documentário": ["realidade", "narrativa", "documentação", "entrevista",
                     "informação", "fatos", "história", "reportagem",
                     "pesquisa", "experiência", "contexto", "narrativa",
                     "revelação", "reflexão", "observação", "experiência",
                     "sociologia", "história", "relato", "análise", "conclusão"],

    "teatro": ["peça", "drama", "narrativa", "atriz", "ator", "cenário",
               "diretor", "espetáculo", "cena", "representação",
               "dramaturgia", "roteiro", "história", "performance",
               "arte", "arte cênica", "ação", "interpretação",
               "dramatização", "cultura", "tradição", "em cena"]
}

# Aqui utilizamos uma variável global para armazenar o texto extraído
texto_extraido = None

# Aqui temos uma função para extrair texto de um PDF
def extrair_texto_pdf(filepath):
    # Aqui temos uma variável chamada texto_pdf para guardar o texto do pdf
    texto_pdf = ""
    with fitz.open(filepath) as pdf:
        # Aqui percorremos as páginas do pdf
        for pagina in pdf:
            # Aqui adicionamos o texto de cada página em uma variável chamada texto_pdf
            texto_pdf += pagina.get_text()
    # Aqui retornamos a nossa variável texto_pdf, a qual contém o texto do nosso pdf
    return texto_pdf

# Aqui temos uma função para identificar o tema do texto
def identificar_tema(texto):
    # Aqui classificamos o texto em um dos temas definidos
    # Aqui dividimos o texto em frases
    frases = nltk.sent_tokenize(texto)
    # Aqui geramos embeddings para as frases
    embeddings = model.encode(frases)

    # Aqui inicializamos as classificações
    classificacoes = {tema: 0 for tema in temas}

    # Aqui verificamos as palavras-chave em cada frase e aumentamos a contagem
    for i, frase in enumerate(frases):
        # Aqui executamos um loop por cada tema
        for tema in temas:
            # Aqui executamos um loop em cada palavra nas palavras_chave do tema
            for palavra in palavras_chave[tema]:
                # Aqui se a palavra(minuscula) estiver na frase(também minúscula)
                if palavra.lower() in frase.lower():
                    # Adicionamos mais 1 ao contador da classificaçoes do tema
                    classificacoes[tema] += 1

    # Aqui retornamos o tema com mais ocorrências(encontra a chave neste caso o tema, com o maior valor no
    # dicionario classificacoes e atribui essa chave à variável tema_classificado
    tema_classificado = max(classificacoes, key=classificacoes.get)
    # Aqui retornamos a variável tema_classificado
    return tema_classificado

# Aqui temos a função para encontrar a resposta usando similaridade de cosseno
def encontrar_resposta(texto, pergunta):
    # Aqui usamos o modulo NLTK para dividir o texto em frases individuais
    trechos = nltk.sent_tokenize(texto)
    # Aqui transformamos cada frases de trechos em um vetor de embeddings(convert_to_tensor=True, converte os
    # embeddings gerados em tensores do pytorch
    embeddings_trechos = model.encode(trechos, convert_to_tensor=True)
    embedding_pergunta = model.encode(pergunta, convert_to_tensor=True)

    # Aqui calculamos a similaridade entre o embedding da pergunta e os embeddings de varios trechos de texto
    similaridades = cosine_similarity(
        # Aqui convertemos o embedding da pergunta em um numpy array, garantindo que ele esteja na cpu e
        # redimensionamos o array para ter uma única linha e várias colunas
        embedding_pergunta.cpu().numpy().reshape(1, -1),
        # Aqui coonvertemos o tensor dos embeddings dos trechos para um numpy array e garantimos que ele esteja na cpu
        embeddings_trechos.cpu().numpy()
    )
    #similaridades retorna uma matriz

    # Aqui encontramos o índice do trecho mais similar à pergunta
    indice_resposta = np.argmax(similaridades)
    # Aqui estabelecemos um treshold para verificar se encontramos a resposta para a pergunta
    if similaridades[0][indice_resposta] < 0.5:
        return "Desculpe, não consegui encontrar uma resposta para sua pergunta."
    # Aqui retornamos o trecho responsável pela resposta
    return trechos[indice_resposta]

# Aqui temos a função para resumir o texto
def resumir_texto(texto):
    # Aqui dividimos o texto em sentenças, igual fizemos em outras partes do código
    trechos = nltk.sent_tokenize(texto)
    # Aqui geramos embeddings para cada trecho
    embeddings_trechos = model.encode(trechos, convert_to_tensor=True)
    # Aqui geramos embeddings para o texto todo
    embedding_texto = model.encode(texto, convert_to_tensor=True)

    # Aqui calculamos a similaridade de cada trecho com o texto completo
    similaridades = cosine_similarity(
        embedding_texto.cpu().numpy().reshape(1, -1),
        embeddings_trechos.cpu().numpy()
    )

    # Aqui Selecionamos as frases mais relevantes para o resumo
    # Aqui retornamos os índices que ordenam o array de similaridade, depois selecionamos os últimos tres indices
    # e por final invertemos a ordem, fazendo com que os índices das sentenças mais relevantes apareçam
    # em ordem decrescente de similaridade
    indices_mais_relevantes = np.argsort(similaridades[0])[-3:][::-1]
    # Aqui criamos um resumo a partir das sentenças mais relevantes
    resumo = " ".join([trechos[i] for i in indices_mais_relevantes])
    # Aqui retornamos o resumo
    return resumo

# Aqui temos a função para carregar o PDF e fazer perguntas
def carregar_pdf():
    #declaramos a variável global texto_extraído
    global texto_extraido
    # Aqui abrimos um explorador de arquivos para selecionar somente arquivos pdf
    filepath = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    # Aqui verificamos se um arquivo foi selecionado, ou seja se a variável filepath possui um valor válido
    if filepath:
        # Aqui se o arquivo foi selecionado eu chamo a função extrair_texto_pdf e salvo o texto extraído na variável
        # texto_extraido
        texto_extraido = extrair_texto_pdf(filepath)
        # Aqui eu printo no console que foi extraído o texto do pdf
        print("Texto extraído do PDF.")

# Aqui temos a função para perguntar
def perguntar():
    # Aqui verificamos se um arquivo pdf foi carregado, caso nao tenha sido emite o aviso de carregar um arquivo pdf
    if texto_extraido is None:
        messagebox.showwarning("Aviso", "Por favor, carregue um PDF antes de fazer uma pergunta.")
        return
    # Aqui capturamos o texto que o usuário digitou em um campo de entrada e armazenamos na variável pergunta
    pergunta = entrada_pergunta.get()
    # Aqui identificamos perguntas fixas, no caso , é o tema extraído
    if pergunta.lower() == 'do que se trata':
        tema = identificar_tema(texto_extraido)
        resposta = f"O tema do texto parece ser sobre: {tema}."
    # Aqui pedimos para resumir o texto
    elif pergunta.lower() == 'resumir o texto':
        resposta = resumir_texto(texto_extraido)
    else:
        # Aqui chamamos a função encontrar_resposta para obter a resposta e salva la na variável resposta
        resposta = encontrar_resposta(texto_extraido, pergunta)
    # Aqui habilitamos o campo de resposta
    texto_resposta.config(state=tk.NORMAL)
    # Aqui limpamos o campo de resposta, quando obtemos uma nova resposta
    texto_resposta.delete(1.0, tk.END)
    # Aqui inserimos a nova resposta
    texto_resposta.insert(tk.END, resposta)
    # Aqui desabilitamos novamente o campo da resposta
    texto_resposta.config(state=tk.DISABLED)

# Interface gráfica
# Aqui inicializamos uma nova instancia da classe Tk, a qual é a janela principal da aplicação
janela = tk.Tk()
# Aqui definimos o título da janela
janela.title("Carregar arquivo .pdf")
# Aqui criamos o botão de carregar pdf e quando ele é clicado ele executa a função carregar_pdf
botao_abrir = tk.Button(janela, text="Carregar PDF", command=carregar_pdf)
# Aqui adicionamos o botão a janela
botao_abrir.pack(pady=20)
# Aqui criammos um campo de entrada de texto para as perguntas
entrada_pergunta = tk.Entry(janela, width=50)
# Aqui adicionamos o campo de entrada de texto a janela
entrada_pergunta.pack(pady=10)
# Aqui criamos um botão para fazer as perguntas e quando ele é clicado , o mesmo executa a função perguntar
botao_perguntar = tk.Button(janela, text="Perguntar", command=perguntar)
# Aqui adicionamos o botão de perguntar a janela
botao_perguntar.pack(pady=5)
# Aqui criamos uma área de texto para exibir a resposta
texto_resposta = tk.Text(janela, height=10, width=50, state=tk.DISABLED)
# Aqui adicionamos a área de texto a janela
texto_resposta.pack(pady=10)
# Aqui iniciamos o loop principal da interface gráfica
janela.mainloop()
