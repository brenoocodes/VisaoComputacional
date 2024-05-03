import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
#inicializando o mediapipe para reconhecimento das mãos
reconhecimento_maos = mp.solutions.hands
desenho_mp = mp.solutions.drawing_utils
maos = reconhecimento_maos.Hands()


if webcam.isOpened():
    while True:
        validacao, frame = webcam.read()
        if validacao:
            #inverter o frame ao longo do eixo x
            frame = cv2.flip(frame, 1)
            #converter o frame de BGR (padrão OpenCv) para RGB (padrão do MediaPipe)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lista_maos = maos.process(frameRGB)
            altura, largura, _ = frame.shape


            # Desenhar linhas e legendar os pixels na tela
            # for y in range(0, altura, 50):
            #     cv2.line(frame, (0, y), (largura, y), (0, 255, 0), 1)  # Desenha linhas horizontais
            #     for x in range(0, largura, 50):
            #         cv2.line(frame, (x, 0), (x, altura), (0, 255, 0), 1)  # Desenha linhas verticais
            #         cv2.putText(frame, f'({x}, {y})', (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)



            #processar o reconhecimentos de mãos
            pontos = []
            #se mãos detectadas no frame
            if lista_maos.multi_hand_landmarks:
                for mao in lista_maos.multi_hand_landmarks:
                    #desenhar os pontos e as conexões das mãos no frame
                    desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)
                    #podemos desenhar os numeros dos pontos também
                    for numero_ponto, coordenada_do_ponto in enumerate(mao.landmark):
                        cx, cy = int(coordenada_do_ponto.x * largura), int(coordenada_do_ponto.y * altura)
                        cv2.putText(frame, f'{numero_ponto}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 20, 20), 1)
                        pontos.append((cx, cy, numero_ponto)) 


                    #fazer o contador de dedos
                    dedos = [8, 12, 16, 20]
                    contador = 0
                    if mao:
                        #fazer a logica para o dedao funcionar tanto na mão esquerda ou direita
                        #mão esquerda
                        if pontos[0][0] < pontos[4][0]:
                            if pontos[4][0] > pontos[3][0]:
                                contador += 1
                        #mão direta
                        if pontos[0][0] > pontos[4][0]:                 
                            if pontos[4][0] < pontos[3][0]:
                                contador += 1
                        
                        #agora a lógica para os quatros dedos restante
                        for posicao_dedos in dedos:
                            #Como estamos nos dedos superiores estamos pegando a posição y do plano, atráves do [1] presente no segundo parênteses e como o plano é de cima para baixo  verificamos se o pontos em cima é menor do que o ponto em baixo, lembrando que o plano cresce para baixo
                            #Ex: se ponto 8 estiver menor que 7 é sinal que ele está acima, então dedo está levantado, caso contrário dedo abaixado
                            if pontos[posicao_dedos][1] < pontos[posicao_dedos - 1 ][1]:
                                contador += 1
                        cv2.putText(frame, str(contador), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)



            cv2.imshow('WebCam', frame)
        fechar = cv2.waitKey(1)
        if fechar == 27 or cv2.getWindowProperty('WebCam', cv2.WND_PROP_VISIBLE) < 1:
            break

webcam.release()
cv2.destroyAllWindows()
