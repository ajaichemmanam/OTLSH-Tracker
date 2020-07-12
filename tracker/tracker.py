from collections import deque

import numpy as np

from matching import embedding_distance, fuse_motion, linear_assignment, gate_cost_matrix, iou_distance
from kalman_filter import KalmanFilter
from log import logger
from utils import mkdir
from track import Track, BaseTrack, TrackState

# LSH
from lshash.lshash import LSHash


def createlshash():
    k = 32  # hash size #12
    L = 8  # number of tables #8
    d = 128  # Dimension of Feature vector
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L, seed=40)
    return lsh


def checklshash(lsh, unmatchedDetections, unmatchedTracks):
    tempList = [(t[0].track_id, t[1]) for t in unmatchedTracks]
    unmatchedTrackIds = dict(tempList)
    matches = []
    for (detection, i) in unmatchedDetections:
        responses = lsh.query(
            detection.curr_feat, num_results=3, distance_func="hamming")
        for idx, response in enumerate(responses):
            distance = response[1]
            feat, trackid = response[0]
            if(trackid in unmatchedTrackIds.keys()):
                print('LSH Results: ', trackid, distance)
                matches.append((unmatchedTrackIds[trackid], i))
                unmatchedDetections.remove((detection, i))
                unmatchedTrackIds.pop(trackid)
                break
            # return response
    u_detection = [d[1] for d in unmatchedDetections]
    u_track = [t for t in unmatchedTrackIds.values()]
    return matches, u_track, u_detection


class Tracker(object):
    def __init__(self, frame_rate=30, lsh_mode=0):
        '''  
        Parameters
        ----------
        frame_rate: int
            Framerate of video tracked
        lsh_mode : int (can have 0, 1, 2 as values)
        0: LSH disabled
        1: LSH Low Impact
        2: LSH Higher Impact
        '''
        # Create LSH Table
        self.lsh_mode = lsh_mode
        if lsh_mode > 0:
            self.lsh = createlshash()
        else:
            self.lsh = None

        self.tracked_tracks = []  # type: list[Track]
        self.lost_tracks = []  # type: list[Track]
        self.removed_tracks = []  # type: list[Track]

        self.frame_id = 0  # Initial Frame ID
        self.det_thresh = 0.0  # Minimum confidence of detection to start new track
        track_buffer = 30  # Prev detections to store
        self.min_box_area = 200  # Minimum Box Area to be considered a valid detection
        # Number of frames to be remembered
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        # Maximum frames to wait for a track to be considered lost
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()  # Kalman Filter

    def updateTracks(self, detections):
        active_tracks = self.update(detections)
        outTracks = []
        for t in active_tracks:
            tlwh = t.tlwh
            # Width/Height > 1.6  True if width >> height else False
            vertical = tlwh[2] / tlwh[3] > 1.6
            # People Generally have height more than width.
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                outTracks.append(t)

        return outTracks

    def update(self, detections=None):
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        if not detections:  # If detections None
            detections = []

        ''' Add newly detected tracklets to tracked_tracks'''
        unconfirmed = []
        tracked_tracks = []  # type: list[Track]
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)

        ''' Matching with embedding'''
        track_pool = join_tracklists(tracked_tracks, self.lost_tracks)
        # Predict location with Kalman Filter
        # for track in track_pool:
        #     track.predict()  #Predict Individually each track
        Track.multi_predict(track_pool)  # Predict Together all Tracks
        # Get Embedding Distance
        dists = embedding_distance(track_pool, detections)
        # dists = gate_cost_matrix(self.kalman_filter, dists, track_pool, detections)
        dists = fuse_motion(
            self.kalman_filter, dists, track_pool, detections)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=0.7)

        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            det = detections[det_idx]
            if self.lsh:
                self.lsh.index(det.curr_feat, track.track_id)  # LSH Indexing

            if track.state == TrackState.Tracked:
                track.update(detections[det_idx], self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)

        '''LSH Similarity - Higher Effect'''

        if self.lsh and self.lsh_mode == 2:
            unmatchedDetections = [(detections[i], i) for i in u_detection]
            unmatchedTracks = [(track_pool[j], j)
                               for j in u_track if track_pool[j].state == TrackState.Tracked]
            matches, u_track, u_detection = checklshash(
                self.lsh, unmatchedDetections, unmatchedTracks)
            for track_idx, det_idx in matches:
                track = [t for t, i in unmatchedTracks if i == track_idx][0]
                det = detections[det_idx]
                self.lsh.index(det.curr_feat, track.track_id)  # LSH Indexing
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks.append(track)

        ''' Matching with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_tracks = [track_pool[i]
                            for i in u_track if track_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_tracks, detections)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=0.5)

        for track_idx, det_idx in matches:
            track = r_tracked_tracks[track_idx]
            det = detections[det_idx]
            if self.lsh:
                self.lsh.index(det.curr_feat, track.track_id)  # LSH Indexing
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracks.append(track)

        '''LSH Similarity - Reduced Effect'''
        if self.lsh and self.lsh_mode == 1:
            unmatchedDetections = [(detections[i], i) for i in u_detection]
            unmatchedTracks = [(r_tracked_tracks[j], j) for j in u_track]
            matches, u_unconfirmed, u_detection = checklshash(
                self.lsh, unmatchedDetections, unmatchedTracks)
            for track_idx, det_idx in matches:
                track = [t for t, i in unmatchedTracks if i == track_idx][0]
                det = detections[det_idx]
                self.lsh.index(det.curr_feat, track.track_id)  # LSH Indexing
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks.append(track)

        # Mark unmatched tracks as lost
        for it in u_track:
            track = r_tracked_tracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        '''New unconfirmed tracks'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=0.7)
        for track_idx, det_idx in matches:
            unconfirmed[track_idx].update(detections[det_idx], self.frame_id)
            activated_tracks.append(unconfirmed[track_idx])

        '''Remove unconfirmed tracks'''
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)

        """ Initialise new tracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_tracks.append(track)

        """ Mark Track as lost"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = join_tracklists(
            self.tracked_tracks, activated_tracks)
        self.tracked_tracks = join_tracklists(
            self.tracked_tracks, refind_tracks)
        self.lost_tracks = remove_from_tracklists(
            self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = remove_from_tracklists(
            self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_tracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks)
        # get scores of lost tracks
        output_tracks = [
            track for track in self.tracked_tracks if track.is_activated]

        logger.debug(
            '-----------Frame No. {}-----------'.format(self.frame_id))
        logger.debug('Active: {}'.format(
            [track.track_id for track in activated_tracks]))
        logger.debug('ReFound: {}'.format(
            [track.track_id for track in refind_tracks]))
        logger.debug('Lost: {}'.format(
            [track.track_id for track in lost_tracks]))
        logger.debug('Deleted: {}'.format(
            [track.track_id for track in removed_tracks]))

        return output_tracks


'''


Joins 2 tracklists without duplication
'''


def join_tracklists(tracklista, tracklistb):
    exists = {}
    res = []
    for t in tracklista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tracklistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


'''
Removes listb tracks from lista
'''


def remove_from_tracklists(tracklista, tracklistb):
    tracks = {}
    for t in tracklista:
        tracks[t.track_id] = t
    for t in tracklistb:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())


'''
Remove duplicate tracks
'''


def remove_duplicate_tracks(tracka, trackb):
    pdist = iou_distance(tracka, trackb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = tracka[p].frame_id - tracka[p].start_frame
        timeq = trackb[q].frame_id - trackb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(tracka) if not i in dupa]
    resb = [t for i, t in enumerate(trackb) if not i in dupb]
    return resa, resb
