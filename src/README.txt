Source code of PDER

Author: Zeyu Li zyli@cs.ucla.edu


==========================
   * preprocessing.py *
==========================

1. Read data from "raw/[name of dataset]/"
2. Extract following informations
    2.1 Uq - Ua
    2.2 Uq - Q
    2.3 Q - Ua
    2.4 Q - Q content
    2.5 Uq, Q, Ua*, Ua'
3. Split it into batches

Here are the steps:
1. From Posts.xml extract Question & Answer pairs
2. Questions -> User ID (2.2)
3. Answers -> User ID
4. Question -> Accepted answer
5.

   - **posts**.xml
       - Id
       - PostTypeId
          - 1: Question
          - 2: Answer
       - ParentID (only present if PostTypeId is 2)
       - AcceptedAnswerId (only present if PostTypeId is 1)
       - CreationDate
       - Score
       - ViewCount
       - Body
       - OwnerUserId
       - LastEditorUserId
       - LastEditorDisplayName="Jeff Atwood"
       - LastEditDate="2009-03-05T22:28:34.823"
       - LastActivityDate="2009-03-11T12:51:01.480"
       - CommunityOwnedDate="2009-03-11T12:51:01.480"
       - ClosedDate="2009-03-11T12:51:01.480"
       - Title=
       - Tags=
       - AnswerCount
       - CommentCount
       - FavoriteCount

