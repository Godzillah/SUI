import numpy
import logging
import matplotlib.pyplot as plt

from .utils import probability_of_successful_attack, sigmoid
from .utils import possible_attacks, effortless_target_areas, get_player_largest_region, get_score_current_player
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand
from .pytorchClasif import *

class AI:
    """Agent using Win Probability Maximization (WPM) using player scores
    This agent estimates win probability given the current state of the game.
    As a feature to describe the state, a vector of players' scores is used.
    The agent choses such moves, that will have the highest improvement in
    the estimated probability.

    This is AI used for testing. Logistic Regression (with use of PyTorch) is used for classification. The possible attack
    with highest proba of class 1 is processed.

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
        self.trained_results = numpy.genfromtxt('./trainFiles/trainingClassesWithImprovement.csv',dtype=int)
        self.trained_vectors = numpy.genfromtxt('./trainFiles/trainingFeaturesWithImprovement.csv',dtype=float, delimiter=",")

        # create model of logistic regression
        self.best_model_full, self.losses_full, self.accuracies_full, self.epochs_list_full = train_all_fea_llr(100, 0.01, 128, self.trained_vectors, self.trained_results)

        # graphs for accuracy and loss development of logistic regression
        '''
        figure = plt.figure(figsize=(10, 10))
        performance_plot = figure.add_subplot(2,1,1)
        performance_plot.plot(self.epochs_list_full, self.accuracies_full, color = "orchid", label="accuracy development")
        performance_plot.set_title('All Features Logistic Regression Performance', fontsize=10)
        performance_plot.set_xlabel('Count of epochs', fontsize=8, horizontalalignment='right', x=1.0)
        performance_plot.legend(prop={'size': 10})

        performance_plot2 = figure.add_subplot(2,1,2)
        performance_plot2.plot(self.epochs_list_full, self.losses_full, color = "indigo", label="loss development")
        performance_plot2.set_xlabel('Count of epochs', fontsize=8, horizontalalignment='right', x=1.0)
        performance_plot2.legend(prop={'size': 10})

        plt.savefig('learn_graph_lr.png') # save graph as png
        '''

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn
        This agent estimates probability to win the game from the feature vector associated
        with the outcome of the move and chooses such that has highest improvement in the
        probability. Feature vectore contains several features of our AI and also oponent,
        on which our AI wants to attack.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        turns = self.possible_turns()
        calculated_features = [] # for adding trained vectors for class prediction

        if (turns):
            for t in turns:
                self.logger.debug("Looking for possible turns.")
                improvement_float = float (t[2]) * 1000

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

                # add all calculated features and create tested vector for Logistic Regression
                calculated_features.append([improvement_float, score_player_value_float, dice_player_value_float, owned_fields_player_float,effortless_target_areas_sum_player_float, largest_region_player_float, score_oponent_value_float, dice_oponent_value_float, owned_fields_oponent_float, effortless_target_areas_sum_oponent_float, largest_region_oponent_float])

            calculated_features_array = numpy.array(calculated_features).astype(numpy.float32)
            self.logger.debug(calculated_features_array)
            prediction = self.best_model_full.prob_class_1(calculated_features_array)

            prediction_list = prediction.tolist()
            self.logger.debug(prediction_list)
            # find the biggest proba of class 1 in all tested vectors and index of this prediction (index of this turn in turns list as well)
            best_prediction = max(prediction_list)
            best_index = prediction_list.index(best_prediction)
            self.logger.debug(best_index)


        if turns:
            turn = turns[best_index] # find value which has biggest proba of class 1
            self.logger.debug("Possible turn: {}".format(turn))

            return BattleCommand(turn[0], turn[1]) # finally attack

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self):
        """Get list of possible turns with the associated improvement
        in estimated win probability
        """
        turns = [] # list for saving filtered possible turns

        features = [] # list for calculated features of our AI

        # get features for player's score, dice, number of owned fields, sum of effortless targets to attack and size of players largest region
        for p in self.players_order:
            score_player_value = get_score_current_player(self.board, p)
            dice_player_value = self.board.get_player_dice(p)
            owned_fields_player = len(self.board.get_player_areas(p))
            effortless_target_areas_sum_player = effortless_target_areas(self.board, p)
            largest_region_player = get_player_largest_region(self.board, p)

            sum_features_player = score_player_value + dice_player_value + owned_fields_player + effortless_target_areas_sum_player + largest_region_player # get sum of features
            features.append(sum_features_player)

        win_probability = numpy.log(sigmoid(numpy.dot(numpy.array(features), self.weights)))


        self.get_largest_region()

        for source, target in possible_attacks(self.board, self.player_name):
            source_name = source.get_name()
            source_power = source.get_dice()

            oponent_name = target.get_owner_name()
            target_power = target.get_dice()

            increase_score = False

            if (source_name in self.largest_region): # increase score if source player has the largest region
                increase_score = True

            successful_attack_probability = probability_of_successful_attack(self.board, source_name, target.get_name())

            if (increase_score or source_power == 8) and (successful_attack_probability > 0.5):
                new_features = [] # list of new features with features of oponent

                for p in self.players_order:
                    index = self.players_order.index(p)

                    if (p == self.player_name):
                        new_features.append(features[index] + 1 if increase_score else features[index])

                    elif (p == oponent_name): # compute features for oponent
                        score_oponent_value = get_score_current_player(self.board, p, skip_area=target.get_name())
                        dice_oponent_value = self.board.get_player_dice(p)
                        owned_fields_oponent = len(self.board.get_player_areas(p))
                        effortless_target_areas_sum_oponent = effortless_target_areas(self.board, p)
                        largest_region_oponent = get_player_largest_region(self.board, p)

                        sum_features_oponent = score_oponent_value + dice_oponent_value + owned_fields_oponent + effortless_target_areas_sum_oponent + largest_region_oponent
                        new_features.append(sum_features_oponent)

                    else:
                        new_features.append(features[index])

                new_win_probability = numpy.log(sigmoid(numpy.dot(numpy.array(new_features), self.weights)))

                # calculate final improvement
                calculated_improvement = new_win_probability - win_probability

                # write neccesary info about turn (source_name, target name, calculated improvement) and also additional info about player, oponent for writing to testing vector
                turns.append([source_name, target.get_name(), calculated_improvement, score_player_value, dice_player_value, owned_fields_player,effortless_target_areas_sum_player, largest_region_player, score_oponent_value, dice_oponent_value, owned_fields_oponent, effortless_target_areas_sum_oponent, largest_region_oponent])

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
