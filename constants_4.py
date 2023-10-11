#* Copyright 2023 The Board of Trustees of the University of Illinois. All Rights Reserved.
#
# * Licensed under the terms of the Apache License 2.0 license (the "License")
#
# * The License is included in the distribution as License.txt file.
#
# * You may not use this file except in compliance with the License.
#
# * Software distributed under the License is distributed on an "AS IS" BASIS,
#
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# * See the License for the specific language governing permissions and limitations under the License.



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Parameters for the Model of V1-based salience
# Developed and written by Rachel F Heaton and John E Hummel
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *



# This code dynamically links Pillow which requires the following information to be included in any redistributions or uses:
#The Python Imaging Library (PIL) is
#
#    Copyright © 1997-2011 by Secret Labs AB
#    Copyright © 1995-2011 by Fredrik Lundh
#
#Pillow is the friendly PIL fork. It is
#
#    Copyright © 2010-2023 by Jeffrey A. Clark (Alex) and contributors.
#
#Like PIL, Pillow is licensed under the open source HPND License:
#
#By obtaining, using, and/or copying this software and/or its associated
#documentation, you agree that you have read, understood, and will comply
#with the following terms and conditions:
#
#Permission to use, copy, modify and distribute this software and its
#documentation for any purpose and without fee is hereby granted,
#provided that the above copyright notice appears in all copies, and that
#both that copyright notice and this permission notice appear in supporting
#documentation, and that the name of Secret Labs AB or the author not be
#used in advertising or publicity pertaining to distribution of the software
#without specific, written prior permission.
#
#SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
#SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
#IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
#INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
#LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
#OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
#PERFORMANCE OF THIS SOFTWARE.

#This code dynamically links to numpy which requires the following information to be included in any redistributions or uses:

#Copyright (c) 2005-2023, NumPy Developers.
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met:
#
#    * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#    * Neither the name of the NumPy Developers nor the names of any
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This code dynamically links MPI which requires the following information to be included in any redistributions or uses:

# =======================
# LICENSE: MPI for Python
# =======================
#
# :Author:       Lisandro Dalcin
# :Contact:      dalcinl@gmail.com
# :Web Site:     http://mpi4py.googlecode.com/
# :Organization: CIMEC <http://www.cimec.org.ar>
# :Address:      CCT CONICET, 3000 Santa Fe, Argentina
#
#
# Copyright (c) 2013, Lisandro Dalcin.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


PHASES = ['sin', 'cos']
NUM_ORIENTATIONS = 8
FREQUENCIES = [6]
V1_MIN_FILTER_THRESHOLD = 0.02

VISUAL_FIELD_RADIUS = 800
BLOB_COLORS = ["red", "green", "blue", "yellow","white", "black"]

# V1 constants
V1_HYPERCOLUMN_SCALES  = (10,20,40)
NEIGHBOR_DISTANCE_THRESHOLD = 0.67 #1.0 # 2.0 # for setting neighbors in V1: the bigger this is, the further away hypercolumns look for neighbors
V1_HYPERCOLUMN_OFFSET       =  0.5 #0.1#0.25 # 0.667 # scaling constant: what fraction of a hypercolumn radius do hypercolumns live apart
CORTICAL_MAGNIFICATION = 0.09

# color channel constants
WB_CHANNEL = 0
RG_CHANNEL = 1
BY_CHANNEL = 2
COLOR_CHANNELS = [WB_CHANNEL,RG_CHANNEL,BY_CHANNEL]

#ToDo: we are tryng to make the filters larger and closer together (hence V1_Hypercolumn_Offset from .67 to .5), but so far it doesn't seem to be working  4/12/19

FILTER_MULTIPLIER           = 15 # 20 # the multiplier on the filter value on converting it to an input to the membrane potential
FILTER_SIGMA = 0.3 #0.25 # changed 12/02/19 # 0.3 # 0.3 # this is the sigma on the gaussian envelope over the filter; was 0.3 prior to 3/1/19

