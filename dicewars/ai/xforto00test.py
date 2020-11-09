import numpy
import logging
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_roc_curve

from .utils import probability_of_successful_attack, sigmoid
from .utils import possible_attacks, effortless_target_areas, get_player_largest_region, get_score_current_player

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand

import matplotlib.pyplot as plt



class AI:
    """Agent using Win Probability Maximization (WPM) using player scores
    This agent estimates win probability given the current state of the game.
    As a feature to describe the state, a vector of players' scores is used.
    The agent choses such moves, that will have the highest improvement in
    the estimated probability.

    This is AI used for training. Feature vectors are extracted and their class
    is predicted based on trained MLP ANN (scikit-learn library). The possible attack
    with highest proba of class 1 is processed.

    TO DO - Use some other type of MLP ANN (e.g. PyTorch), work with scikit-learn is
    really simple and I don't know whether it is allowed.
    """
    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game
        Attributes
        ----------
        players_order : list of int
            Names of players in the order they are playing, with the agent being first
        weights : dict of numpy.array
            Weights for estimating win probability
        largest_region: list of int
            Names of areas in the largest region
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.players = board.nb_players_alive()

        self.largest_region = []

        self.players_order = players_order
        while self.player_name != self.players_order[0]:
            self.players_order.append(self.players_order.pop(0))

        mu, sigma = 0, 1 # mean and standard deviation, randomly init weights of ANN

        self.weights = {
            2: numpy.random.normal(mu, sigma, size=(2)),
            3: numpy.random.normal(mu, sigma, size=(3)),
            4: numpy.random.normal(mu, sigma, size=(4)),
            5: numpy.random.normal(mu, sigma, size=(5)),
            6: numpy.random.normal(mu, sigma, size=(6)),
            7: numpy.random.normal(mu, sigma, size=(7)),
            8: numpy.random.normal(mu, sigma, size=(8)),
        }[self.players]

        # generate trained vectors and their classes from csv files
        self.trained_results = numpy.genfromtxt('./trainFiles/trainedClasses.csv',dtype=int)
        self.trained_vectors = numpy.genfromtxt('./trainFiles/trainedImprovements.csv',dtype=float, delimiter=",")

        # do preprocessing of trained_vectors (more suitable for fitting MLP)
        self.trained_vectors_preprocessed = preprocessing.scale(self.trained_vectors)

        # init and train MLP MLPClassifier with vectors for training (extracted in xforto00 AI)
        self.clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(35, 35, 35), max_iter=5000000000)
        self.clf.fit(self.trained_vectors_preprocessed, self.trained_results) # train SVM with trained vectors and their results
        ax = plt.gca()
        svc_disp = plot_roc_curve(self.clf, self.trained_vectors_preprocessed, self.trained_results)
        svc_disp.plot(ax=ax, alpha=1e-5)
        plt.savefig('learn_graph.png')

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn
        This agent estimates probability to win the game from the feature vector associated
        with the outcome of the move and chooses such that has highest improvement in the
        probability.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        turns = self.possible_turns()
        calculated_improvements = []
        calculated_features = [] # for adding trained vectors for class prediction

        if (turns):
            for t in turns:
                self.logger.debug("Looking for possible turns.")
                improvement_float = float (t[2]) * 1000000
                calculated_improvements.append(improvement_float)

                score_player_value_float = float (t[3])
                dice_player_value_float = float (t[4])
                owned_fields_player_float = float (t[5])
                effortless_target_areas_sum_player_float = float (t[6])
                largest_region_player_float = float (t[7])

                score_oponent_value_float = float (t[8])
                dice_oponent_value_float = float (t[9])
                owned_fields_oponent_float = float (t[10])
                effortless_target_areas_sum_oponent_float = float (t[11])
                largest_region_oponent_float = float (t[12])

                # add all calculated features and create tested vector for MLP
                calculated_features.append([score_player_value_float, dice_player_value_float, owned_fields_player_float,effortless_target_areas_sum_player_float, largest_region_player_float, score_oponent_value_float, dice_oponent_value_float, owned_fields_oponent_float, effortless_target_areas_sum_oponent_float, largest_region_oponent_float])

            # do preprocessing also for trained vectors for better fitting to MLP
            tested_vectors_preprocessed = preprocessing.scale(calculated_features)

            prediction = self.clf.predict_proba(tested_vectors_preprocessed) # predict results of tested vectors
            self.logger.debug(prediction)
            # get only proba of positive class
            predicted_positive_class_proba = prediction[:,1]
            prediction_list = predicted_positive_class_proba.tolist()
            # find the biggest proba of class 1 in all tested vectors and index of this prediction (index of this turn in turns list as well)
            best_prediction = max(prediction_list)
            best_index = prediction_list.index(best_prediction)


        if turns:
            turn = turns[best_index]
            self.logger.debug("Possible turn: {}".format(turn))

            return BattleCommand(turn[0], turn[1]) # finally attack

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self):
        """Get list of possible turns with the associated improvement
        in estimated win probability
        """
        turns = []

        features = []

        # get features for player's score, dice, number of owned fields, sum of effortless targets to attack
        for p in self.players_order:
            score_player_value = get_score_current_player(self.board, p)
            dice_player_value = self.board.get_player_dice(p)
            owned_fields_player = len(self.board.get_player_areas(p))
            effortless_target_areas_sum_player = effortless_target_areas(self.board, p)
            largest_region_player = get_player_largest_region(self.board, p)

            sum_features_player = score_player_value + dice_player_value + owned_fields_player + effortless_target_areas_sum_player + largest_region_player # get sum of features
            features.append(sum_features_player)

        win_prob = numpy.log(sigmoid(numpy.dot(numpy.array(features), self.weights)))


        self.get_largest_region()

        for source, target in possible_attacks(self.board, self.player_name):
            area_name = source.get_name()
            atk_power = source.get_dice()

            target_power = target.get_dice()

            opponent_name = target.get_owner_name()

            increase_score = False

            if area_name in self.largest_region: # increase score if actual player has the largest region
                increase_score = True

            atk_prob = probability_of_successful_attack(self.board, area_name, target.get_name())

            if (increase_score or atk_power == 8) and (atk_prob > 0.3):
                new_features = []
                for p in self.players_order:
                    idx = self.players_order.index(p)

                    if p == self.player_name:
                        new_features.append(features[idx] + 1 if increase_score else features[idx])

                    elif p == opponent_name: # compute features for oponent
                        score_oponent_value = get_score_current_player(self.board, p, skip_area=target.get_name())
                        dice_oponent_value = self.board.get_player_dice(p)
                        owned_fields_oponent = len(self.board.get_player_areas(p))
                        effortless_target_areas_sum_oponent = effortless_target_areas(self.board, p)
                        largest_region_oponent = get_player_largest_region(self.board, p)

                        sum_features_oponent = score_oponent_value + dice_oponent_value + owned_fields_oponent + effortless_target_areas_sum_oponent + largest_region_oponent
                        new_features.append(sum_features_oponent)

                    else:
                        new_features.append(features[idx])

                new_win_prob = numpy.log(sigmoid(numpy.dot(numpy.array(new_features), self.weights)))

                improvement = new_win_prob - win_prob

                #if improvement > -1:
                    # write neccesary info about turn (area_name, target name, calculated improvement) and also additional info about player, oponent for writing to testing vector
                turns.append([area_name, target.get_name(), improvement, score_player_value, dice_player_value, owned_fields_player,effortless_target_areas_sum_player, largest_region_player, score_oponent_value, dice_oponent_value, owned_fields_oponent, effortless_target_areas_sum_oponent, largest_region_oponent])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)

    def get_score_by_player(self, player_name, skip_area=None):
        """Get score of a player
        Parameters
        ----------
        player_name : int
        skip_area : int
            Name of an area to be excluded from the calculation
        Returns
        -------
        int
            score of the player
        """
        players_regions = self.board.get_players_regions(self.player_name, skip_area=skip_area)
        max_region_size = max(len(region) for region in players_regions)

        return max_region_size

    def get_largest_region(self):
        """Get size of the largest region, including the areas within
        Attributes
        ----------
        largest_region : list of int
            Names of areas in the largest region
        Returns
        -------
        int
            Number of areas in the largest region
        """
        self.largest_region = []

        players_regions = self.board.get_players_regions(self.player_name)
        max_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

        for region in max_sized_regions:
            for area in region:
                self.largest_region.append(area)
        return max_region_size
