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

            #processar o reconhecimentos de mãos
            lista_maos = maos.process(frameRGB)
            altura, largura, _ = frame.shape
            #se mãos detectadas no frame
            if lista_maos.multi_hand_landmarks:
                for mao in lista_maos.multi_hand_landmarks:
                    #desenhar os pontos e as conexões das mãos no frame
                    desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)
                    #podemos desenhar os numeros dos pontos também
                    for numero_ponto, coordenada_do_ponto in enumerate(mao.landmark):
                        cx, cy = int(coordenada_do_ponto.x * largura), int(coordenada_do_ponto.y * altura)
                        cv2.putText(frame, f'{numero_ponto}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


            cv2.imshow('WebCam', frame)
        fechar = cv2.waitKey(1)
        if fechar == 27 or cv2.getWindowProperty('WebCam', cv2.WND_PROP_VISIBLE) < 1:
            break

webcam.release()
cv2.destroyAllWindows()