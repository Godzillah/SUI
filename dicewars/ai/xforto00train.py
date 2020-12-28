import numpy
import logging

from .utils import probability_of_successful_attack, sigmoid
from .utils import possible_attacks, effortless_target_areas, get_player_largest_region, get_score_current_player

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    """Agent using Win Probability Maximization (WPM) using player scores

    This agent estimates win probability given the current state of the game.
    As a feature to describe the state, a vector of players' scores is used.
    The agent choses such moves, that will have the highest improvement in
    the estimated probability.

    This is AI used for training and getting trained vectors for MLP ANN for xforto00test,
    normally trained on tournament for 4 players, 50 games.

    Classes are 0 or 1.

    For each processed attack we decide in next round, whether area on which we attacked
    is in our regions or some oponent took it from us.

    Class 1 - Area is still in our regions.
    Class 0 - Area was won by someone else.
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

        self.processed_turns_targets = [] # list for saving targets on which we are decided to attack

        # open files for writing trained feature vectors of attacks and class whether this attack helped us or not
        # paths to val dataset or train dataset - depends whether we extract features for train or val dataset
        self.f = open("./dicewars/ai/xforto00/valFiles/validationClassesWithImprovement.csv","a")
        self.g = open("./dicewars/ai/xforto00/valFiles/validationFeaturesWithImprovement.csv","a")
    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        This agent estimates probability to win the game from the feature vector associated
        with the outcome of the move and chooses such that has highest improvement in the
        probability.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        turns = self.possible_turns()

        if turns:
            #print(turns)
            turn = turns[0]
            self.logger.debug("Possible turn: {}".format(turn))

            owned_fields_ai = self.board.get_player_areas(self.player_name)
            self.logger.debug("Printing owned field of our AI")
            self.logger.debug(owned_fields_ai)
            owned_fields_ai_names = []

            if (len(owned_fields_ai) > 0):
                for owned_field in owned_fields_ai:
                    self.logger.debug(owned_field.get_name())
                    owned_fields_ai_names.append(owned_field.get_name())

            self.logger.debug("Printing processed turns targets")
            self.logger.debug(self.processed_turns_targets)

            # check whether helped previous processed attack and attacked target is still in our areas in our next round
            if (len(self.processed_turns_targets) > 0):
                improvement_float = float (turn[2]) * 1000 # multiply with constant, because improvement could be very small
                score_player_value_float = float (turn[3])
                dice_player_value_float = float (turn[4])
                owned_fields_player_float = float (turn[5])
                effortless_target_areas_sum_player_float = float (turn[6])
                largest_region_player_float = float (turn[7])

                score_oponent_value_float = float (turn[8])
                dice_oponent_value_float = float (turn[9])
                owned_fields_oponent_float = float (turn[10])
                effortless_target_areas_sum_oponent_float = float (turn[11])
                largest_region_oponent_float = float (turn[12])

                # write feature vector to file of trained vectors
                self.g.write(str(improvement_float) + ", " + str(score_player_value_float) + ", " + str(dice_player_value_float) + ", " + str(owned_fields_player_float) + ", " + str(effortless_target_areas_sum_player_float) + ", " +  str(largest_region_player_float) + ", " + str(score_oponent_value_float) + ", " + str(dice_oponent_value_float) + ", " + str(owned_fields_oponent_float) + ", " + str(effortless_target_areas_sum_oponent_float) + ", " + str(largest_region_oponent_float) + "\n")
                if (self.processed_turns_targets[-1] in owned_fields_ai_names):
                    self.logger.debug("Attack in previous round helped us.")
                    self.f.write("1" + "\n")
                else:
                    self.logger.debug("Attack in previous round didnt help us.")
                    self.f.write("0" + "\n")


            # save new attack which we are ready to process to list
            self.processed_turns_targets.append(turn[1])

            return BattleCommand(turn[0], turn[1]) # we are attacking in this round

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

            if (increase_score or atk_power == 8) and (atk_prob > 0.5):
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
                    # write neccesary info about turn (area_name, target name, calculated improvement) and also additional info about player, oponent for writing to training vector
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
