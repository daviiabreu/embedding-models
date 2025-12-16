# Como rodar a aplicação

&emsp; Para buildar a aplicação a partir do docker file use o comando:

```bash

docker build -t stt_tts:latest .

```

&emsp; Então, para rodar a aplicação e alterar os arquivos, use:

```bash

docker run -it --rm -v $(pwd):/usr/src/stt_tts stt_tts:latest /bin/bash

```

&emsp; Isso linkará os arquivos no seu diretório atual com os arquivos da imagem. Além disso, abrirá um bash em que você pode rodar comandos nos arquivos da imagem.
&emsp; Após as alterações, rode o comando:

```bash

python3 pip freeze > requirements.txt && \
exit

```

&emsp; E commite.
