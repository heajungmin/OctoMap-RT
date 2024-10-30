/*
 * Copyright 2024 The Ewha Womans University.
 * All Rights Reserved.
 *
 * Permission to use, copy, modify OR distribute this software and its
 * documentation for educational, research, and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice and the following three paragraphs appear in all
 * copies.
 *
 * IN NO EVENT SHALL THE EWHA WOMANS UNIVERSITY BE
 * LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
 * CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
 * USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE EWHA WOMANS UNIVERSITY
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 *
 * THE EWHA WOMANS UNIVERSITY SPECIFICALLY DISCLAIM ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
 * PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE EWHA WOMANS UNIVERSITY
 * HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 *
 * The authors may be contacted via:
 *
 * Mail:        Heajung Min, Kyung Min Han, and Young J. Kim
 *              Computer Graphics Lab                       
 *              Department of Computer Science and Engineering
 *              Ewha Womans University
 *              11-1 Daehyun-Dong Seodaemun-gu, Seoul, Korea 120-750
 *
 * Phone:       +82-2-3277-6798
 *
 * EMail:       hjmin@ewha.ac.kr
 *              hankm@ewha.ac.kr
 *              kimy@ewha.ac.kr
 */

#include <octomapRT/octomapRT.h>

using namespace octomap;
using namespace octomath;

void printUsage(char *self) {
    std::cerr << "USAGE: " << self << " [options]\n\n";

    std::cerr << "OPTIONS:\n  -i <InputFileName.graph> (required)\n"
                 "  -o <OutputFileName> (required) \n"
                 "  -res <resolution> (optional, default: 0.1 m)\n"
                 "  -m <maxRange> (optional, default: 0.0 m)\n"
                 "\n";
    exit(0);
}

int main(int argc, char *argv[]) {
    try
    {
        timeval start;
        timeval stop;

        // default values:
        double res = 0.1;
        std::string graphFilename = "";
        std::string treeFilename = "";
        double maxRange = -1;

        // get default sensor model values:
        OcTree emptyTree(0.1);
        double clampingMin = emptyTree.getClampingThresMin();
        double clampingMax = emptyTree.getClampingThresMax();
        double probMiss = emptyTree.getProbMiss();
        double probHit = emptyTree.getProbHit();

        int arg = 0;
        while (++arg < argc) {
            if (!strcmp(argv[arg], "-i"))
                graphFilename = std::string(argv[++arg]);
            else if (!strcmp(argv[arg], "-o"))
                treeFilename = std::string(argv[++arg]);
            else if (!strcmp(argv[arg], "-res") && argc - arg < 2)
                printUsage(argv[0]);
            else if (!strcmp(argv[arg], "-res"))
                res = atof(argv[++arg]);
            else if (!strcmp(argv[arg], "-m"))
                maxRange = atof(argv[++arg]);
            else {
                printUsage(argv[0]);
            }
        }

        std::string treeFilenameOT = treeFilename + ".ot";
        std::string treeFilenameBT = treeFilename + ".bt";

        ScanGraph *graph = new ScanGraph();
        if (!graph->readBinary(graphFilename)) {
            std::cout << "error. no input file." << std::endl;
            exit(2);
        }

        gettimeofday(&start, NULL);

        OctomapRT octomapRT(res, maxRange, clampingMin, clampingMax, probHit, probMiss);
        octomapRT.initialize(graph);

        size_t m_currentScan = 1;
        for (ScanGraph::iterator scan_it = graph->begin(); scan_it != graph->end(); scan_it++) {
            octomapRT.insertPointCloud(scan_it);

            m_currentScan++;
        }        
        octomapRT.sync();

        gettimeofday(&stop, NULL);
        double time_to_buildMap = (stop.tv_sec - start.tv_sec) + 1.0e-6 * (stop.tv_usec - start.tv_usec);
        std::cout << "time to build map: " << time_to_buildMap << " sec" << std::endl;

        octomapRT.getTree()->write(treeFilenameOT);
        octomapRT.getTree()->writeBinary(treeFilenameBT);

        octomapRT.cleanup();        
    } catch (std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

