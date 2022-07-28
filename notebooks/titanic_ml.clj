;; # Exploring Titanic Dataset using Clay
;;
;; In this notebook, I am exploring the titanic dataset using clojure tools. After trying out a few different notebooks, I liked Clay the best
;; given its integration with Portal allowing a very simple debugging of data. And the fact that I can write markdown within comments and the they
;; get picked up as literate documents for the namespace is really encouraging. I, honestly, found it difficult to the samething with python and
;; jupyter although the entire community is super comfortable doing just that. Anyway, here goes.
;;
;;## Setup
;;
(ns titanic-ml
  (:require [scicloj.clay.v1.api :as clay]
            [scicloj.clay.v1.tools :as tools]
            [scicloj.clay.v1.extensions :as ext]
            [scicloj.clay.v1.tool.scittle :as scittle]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [scicloj.ml.dataset :as ds]
            [tech.v3.datatype.functional :as funs]
            [scicloj.viz.api :as viz]
            [tablecloth.api :as tc]
            [scicloj.kindly.v2.api :as kindly]
            [scicloj.kindly.v2.kind :as kind]))

^:kind/hidden
(def config {:tools [tools/clerk
                     tools/portal
                     tools/scittle]
             :extensions [ext/dataset]})

^:kind/hidden
(clay/start! config)

^:kind/hidden
(comment
  (scittle/show-doc! "notebooks/titanic_ml.clj")
  (scittle/write-html! "docs/titanic-basic.html"))

;; ## Getting the data
;;
;; I have the titanic dataset downloaded into the datasets folder. Let's just use tablecloth to read in the data.
(def titanicds (tc/dataset "datasets/titanic.csv" {:key-fn keyword}))

;; I could actually use the scicloj.ml.dataset namespace to read in the CSV. It uses tablecloth as well, from what I understand. But I just have
;; this code that I wrote earlier, so am sticking with it. Let's preview the data as a table.
;;
(-> {:column-names (tc/column-names titanicds)
     :row-vectors (tc/rows titanicds :row-vectors)}
    (kindly/consider kind/table))

;; To understand the dataset, we need to also see some meta-information about the dataset. So let's get the dataset info and view that as well.
(def titanicdsinfo (tc/info titanicds))

(-> {:column-names (tc/column-names titanicdsinfo)
     :row-vectors (tc/rows titanicdsinfo :row-vectors)}
    (kindly/consider kind/table))

;; Good, that is quite presentable and easy to understand what is happening as well. Now, let's proceed to implement a simple logistic regression
;; with this data to predict the Survivors field.
;;
;; ## Preprocessing
;;

;; ### Clean Up the Dataset
;;
;; For the titanic dataset, and for this preliminary exploration - where my target is to understand how to work with
;; scicloj ml libraries rather than to understand the data - I am going to drop a few columns and remove some missing values.
;;
;; * I don't want to select the PassengerID col, since that's an identifier. It doesn't add any value to the training itself.
;; * Embarked field has only two missing values. I'd drop the rows rather than impute - just to keep it simple.
;; * Cabin field has quite a large number of missing values. Its not useful to impute > 50% of the data. We will only compromise the data. So dropped.
;; * Name and Ticket field are actually useful if used along side SibSp, Parch fields. But for this run, am dropping them as well.
;;
;; Writing this up as a pipeline, this is what the code looks like.

(def cleanup-ds
  (ml/pipeline
   (mm/select-columns [:Sex :Embarked :Survived :Pclass :Age :Cabin :Name :Ticket :Parch :SibSp])
   (mm/drop-missing [:Embarked])
   (mm/drop-columns [:Cabin :Name :Ticket])))

;; ### Prepare the Dataset
;;
;; Next step is to impute any meaningful missing values, encode the categorical variables such that the dataset can run through a learning
;; algorithm.
;; To do this:
;; * Age is a simple numeric and is usually OK to replace it with the mean age. We could, of course, take advantage of the other columns to arrive
;; at a more meaningful impute. That will be an interesting thing to try in the next iteration.
;; * Sex, Embarked are categorical variables. So let's just one-hot encode them.
;; * Survived, Parch, Pclass and SibSp are all numerical vars and let's treat them as such.
;;
;; Writing this up as a pipeline, this is what the code looks like.

(def prepare-ds
  (ml/pipeline
   (mm/categorical->one-hot [:Sex :Embarked])
   (mm/categorical->number [:Survived :Parch :Pclass :SibSp])
   (mm/replace-missing :Age :value funs/mean)))

;; ## Logistic Regression
;;
;; Next step is to implement a simple pipeline to run a learning algorithm. In this case, I am choosing a logistic regression model with all
;; defaults. We also need to identify the target variable. Lastly, metamorph requires us to specific a id special variable as well. Let's
;; combine these into a pipeline fn as well.
;;
;; This is what the pipeline function would look like.
(def model-fn
  (ml/pipeline
   (mm/set-inference-target :Survived)
   {:metamorph/id :model}
   (mm/model {:model-type :smile.classification/logistic-regression})))

;; Next, we split the dataset into training and test sets.
(def train-test-splits (ds/split->seq titanicds :holdout))

;; Then we combine all the pipeline functions to run on the data as a single vector. Ideally, we would write multiple model-fns and combine them
;; such that we have more pipelines to explore or try. For this example, am sticking to a single pipeline. In a next notebook, extending from this
;; one, I'll add more pipelines and simultaneous exploration like grid search.
(def all-pipelines [(ml/pipeline
                     cleanup-ds
                     prepare-ds
                     model-fn)])

;; Scicloj allows us to evaluate the pipelines in a single go while handling all the details. Of course, this hides all the details from us. We
;; could write it out as independant fit, predict and measure functions. For this example, there isn't much to gain from doing all that. Having said
;; that, it is advantageous to explore that as well. Let's do that in the next section.
;;
;;Here is what a composition of this would look like.
(def eval-results (ml/evaluate-pipelines all-pipelines
                                         train-test-splits
                                         ml/classification-accuracy
                                         :accuracy))

;; We can extract the train and transform metrics from the `eval-results` value. Here is what it looks like.
^:kind/hiccup
[:table
 [:thead
  [:tr
   [:th "Metric"]
   [:th "Train"]
   [:th "Test"]]]
 [:tbody
  [:tr
   [:td :accuracy]
   [:td (:metric (:test-transform (first (first eval-results))))]
   [:td (:metric (:train-transform (first (first eval-results))))]]]]

;; Now, now. That's not bad at all! Let's try to run the fit, predict and measure functions independantly.
;;
;; First, let us construct a full pipeline to process the data with.
(def pipeline-fn
  (ml/pipeline
   cleanup-ds
   prepare-ds
   model-fn))

;; Then, we build the training context. The key element here is we are building a map with pipeline function and metamorph special keys for
;; data and the stage of learning. First is training, so use training dataset and use the `:fit` mode.
(def train-ctx
  (pipeline-fn
  {:metamorph/data (:train (first train-test-splits))
   :metamorph/mode :fit}))

;; Now the training context is built, let's build the test context. It's the same as earlier, but we reassociate mode and date to `:transform` and
;; test dataset.
(def test-ctx
  (pipeline-fn
   (assoc train-ctx
          :metamorph/data (:test (first train-test-splits))
          :metamorph/mode :transform)))


;; So here is the predictions for this case.
(def predictions (-> test-ctx :metamorph/data
                     (ds/column-values->categorical :Survived)))
(def actuals (ds/column (:test (first train-test-splits)) :Survived))

;; Let's compare this against the actuals to see how we performed.
^:kind/hiccup
[:div
 [:table
  [:thead
   [:tr
    [:th "Metric"]
    [:th "Test"]]]
  [:tbody
   [:tr
    [:td :accuracy]
    [:td (ml/classification-accuracy actuals predictions)]]]]]

;; That's it. We got a similar score as the original one and the first one is far simpler to write :smile:
;;
;; ## Closing Notes
;;
;; It took me a litlle more than a day to figure out how to write this code. I did take quite a bit from the scicloj.ml user guides. I guess that's
;; acceptable for a start. I'll add another notebook enhancing this stuff in the coming weeks. I hope to explore more models, more data processing
;; techniques and more clojure tricks along the way.
;;
;; I also think I like Clay as my notebook as default. It allows me to code using emacs. While I did face some hiccup setting up clay and portal to
;; work with emacs - largely due to doom emacs, imo - the final result is perfectly acceptable.
;;
;; Of course, there are bound to be many bugs resulting from my misunderstanding clojure and scicloj.ml. Please do highight those, if you spot any.
;; I will correct the notebooks and generate the HTMLs accordingly.
