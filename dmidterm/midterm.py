import time
import numpy as np
import cv2
from sklearn import svm
import os
import random
import joblib
from sklearn.model_selection import train_test_split
from skimage.feature import hog

path_prefix = './dmidterm/assets/'
model_path = './dmidterm/model.pkl'


def show_img(*images):
    if not isinstance(images, list):
        images = [images]

    for image in images[0]:
        cv2.imshow('', image)
        cv2.waitKey(0)


def get_clf_model(kernel_: str, x_train, y_train, c: int, gamma_: str):
    # create model
    clf = svm.SVC(kernel=kernel_, C=c, gamma=gamma_)

    # train model
    clf.fit(x_train, y_train)

    # get the trained model
    return clf


def get_data(from_: str, hand_type: str) -> list:
    # get all file name in target dir.
    whole_path = path_prefix + from_ + '/' + hand_type + '/'
    names = [f for f in os.listdir(whole_path)]
    return [cv2.imread(whole_path + name) for name in names]


def get_single_validation_img(hand_type: str):
    return random.choice(get_data('test', hand_type))


def convert_to_gray(images: list) -> list:
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]


def create_target_list(amount: int, content: str) -> list:
    return [content for _ in range(amount)]


def get_attributes_hog(images: list):
    hogs = []
    for img in images:
        fd = hog(
            img,
            orientations=8,
            pixels_per_cell=(3, 3),
            cells_per_block=(1, 1),
        )
        hogs.append(fd)
    return hogs


def judge_game_result(we: str, he: str) -> str:
    if we == he:
        return '🟡 Tie (￣ε￣)~♪'
    all_types = ['paper', 'rock', 'scissors']
    we, he = all_types.index(we), all_types.index(he)
    gap = (he - we + 3) % 3
    return '🟢 Win (ﾉ>ω<)ﾉ' if gap - 1 == 0 else '🔴 Lose (≧Д≦)'


def add_hand_emoji(hand: str) -> str:
    if hand == 'paper':
        return hand + ' 🖐'
    if hand == 'rock':
        return hand + ' ✊'
    if hand == 'scissors':
        return hand + ' ✌️'


def start_validation(model):
    try:
        player_chosen_hand_type = input('please tell me what is your hand (paper=5|rock=0|scissors=2): ')
        if player_chosen_hand_type.isdigit() and player_chosen_hand_type in ('5', '2', '0'):
            real_meaning_dict = {'5': 'paper', '0': 'rock', '2': 'scissors'}
            player_chosen_hand_type = real_meaning_dict[player_chosen_hand_type]
    except KeyboardInterrupt:
        print()
        return

    if player_chosen_hand_type not in ['paper', 'rock', 'scissors']:
        print('please provide a correct hand type!\n')
        return

    player_random_hand_img = get_single_validation_img(player_chosen_hand_type)
    player_img_data = get_attributes_hog([player_random_hand_img])[0]
    expected_player_img_target = model.predict([player_img_data])

    print('player choose', add_hand_emoji(player_chosen_hand_type))
    player_chosen_hand_type_window_name = 'player chosen ({})'.format(player_chosen_hand_type)
    cv2.imshow(player_chosen_hand_type_window_name, player_random_hand_img)
    cv2.moveWindow(player_chosen_hand_type_window_name, 0, 0)

    random.seed(time.time())
    npc_chosen_hand_type = random.choice(['paper', 'rock', 'scissors'])
    npc_random_hand_img = get_single_validation_img(npc_chosen_hand_type)
    npc_img_data = get_attributes_hog([npc_random_hand_img])[0]
    expected_npc_img_target = model.predict([npc_img_data])

    print('npc    choose', add_hand_emoji(npc_chosen_hand_type))
    npc_chosen_hand_type_window_name = 'npc chosen ({})'.format(npc_chosen_hand_type)
    cv2.imshow(npc_chosen_hand_type_window_name, npc_random_hand_img)
    cv2.moveWindow(npc_chosen_hand_type_window_name, 600, 0)

    game_result = judge_game_result(expected_player_img_target, expected_npc_img_target)
    game_result_img = cv2.resize(cv2.imread(path_prefix + 'judge/{}.png'.format(game_result[0])), (300, 300))
    cv2.imshow('result', np.concatenate((player_random_hand_img, game_result_img, npc_random_hand_img), axis=1))
    cv2.moveWindow('result', 0, 350)

    print('The expected result is:', game_result)
    print()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_img(img):
    return cv2.resize(img, (300, 300))


def run():
    # detect if model trained.
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        model = None

    if model is None:
        # get data for training.
        train_paper_data = convert_to_gray(get_data('train', 'paper'))
        train_rock_data = convert_to_gray(get_data('train', 'rock'))
        train_scissors_data = convert_to_gray(get_data('train', 'scissors'))

        # get data for testing.
        test_paper_data = convert_to_gray(get_data('test', 'paper'))
        test_rock_data = convert_to_gray(get_data('test', 'rock'))
        test_scissors_data = convert_to_gray(get_data('test', 'scissors'))

        # mark targets related to datasets
        train_paper_target = create_target_list(len(train_paper_data), 'paper')
        train_rock_target = create_target_list(len(train_rock_data), 'rock')
        train_scissors_target = create_target_list(len(train_scissors_data), 'scissors')
        test_paper_target = create_target_list(len(test_paper_data), 'paper')
        test_rock_target = create_target_list(len(test_rock_data), 'rock')
        test_scissors_target = create_target_list(len(test_scissors_data), 'scissors')

        # combine train data
        train_data_origin = train_paper_data + train_rock_data + train_scissors_data

        # combine train target
        train_target_origin = train_paper_target + train_rock_target + train_scissors_target

        # combine test data
        test_data_origin = test_paper_data + test_rock_data + test_scissors_data

        # combine test target
        test_target_origin = test_paper_target + test_rock_target + test_scissors_target

        # HOG
        train_data = get_attributes_hog(train_data_origin)
        test_data = get_attributes_hog(test_data_origin)

        # split data
        result = train_test_split(
            np.concatenate((train_data, test_data)),
            np.array(train_target_origin + test_target_origin),
            test_size=0.2, random_state=0
        )

        # get trained model
        model = get_clf_model('linear', result[0], result[2], 1, 'auto')
        joblib.dump(model, model_path)
        print(model.score(result[1], result[3]))

    start_validation(model)
