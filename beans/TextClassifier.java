package beans;

import java.util.Collection;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeSet;

public class TextClassifier<K> {
        
    Map<K, Integer> countDocumentByCategory = new HashMap<>();
    
    // calculate features
    Map<K, CategoryWords> features = new HashMap<>();
            
    // vocabulary of all documents belongs to the training set
    Set<String> vocabulary = new TreeSet<>();    
    
    Set<K> categorySet = new TreeSet<>();
    
    int trainingSetCapacity;
    
    ThrowsWhenIllegalFeatureException throwsWhenIllegalItemException;    
    
    public TextClassifier() {
        this(new ThrowsWhenIllegalFeatureException() {
            @Override public void when(boolean occurs) {
                if (occurs) {
                    throw new IllegalArgumentException("Illegal feature");
                }                
            }
        });
    }
    
    public TextClassifier(ThrowsWhenIllegalFeatureException exception) {
        this.throwsWhenIllegalItemException = exception;
    }
    
    public void learn(K category, Collection<String> words) {
        ++trainingSetCapacity;
        increaseCategoryDocs(category);
        
        categorySet.add(category);
        
        // update vocabulary
        vocabulary.addAll(words);         
        
        CategoryWords categoryWords = features.get(category);
        if (categoryWords == null) {
            features.put(category, new CategoryWords(category));
        }
        features.get(category).addAll(words);
    }   
    
    public Map<K, Double> probabilities(Collection<String> words) {
        if (words.isEmpty()) {
            throw new IllegalArgumentException("Collection of words must be non-empty");
        }
        UsefulTreeSet<String> wordSet = new UsefulTreeSet<>();
        wordSet.addAdd(words);
        Map<K, Double> result = new HashMap<>();
        for(K c: categorySet) {
            double probability = p(c);
            for(String w: wordSet.list()) {
                probability *= Math.pow(p(w, c), wordSet.getOccurrencesOfItem(w));
            }
            result.put(c, probability);
        }
        return result;
    }
    
    public double p(String w, K category) { 
        return 1.0*(features.get(category).getWordOccurrences(w) + 1)/(vocabulary.size() + features.get(category).getWordSet().getAddedItems());
    }
    
    public double p(K category) {
        return countDocumentByCategory.get(category)*1.0/trainingSetCapacity;
    }
    
    private void increaseCategoryDocs(K category) {
        // total documents in this category
        int numDocs = countDocumentByCategory.getOrDefault(category, 0);
        countDocumentByCategory.put(category, ++numDocs); 
    }    
    
    class CategoryWords<K> {
        
        protected K category;
        
        // A word appears many times on this category? 
        // We use UsefulTreeSet
        protected UsefulTreeSet<String> wordSet = new UsefulTreeSet<>();
        
        public CategoryWords(K category) {
            this.category = category;         
        }
        
        public void addAll(Collection<String> words) {
            wordSet.addAdd(words);
        }
        
        public int getWordOccurrences(String w) {
            return wordSet.getOccurrencesOfItem(w);
        }
        
        public UsefulTreeSet getWordSet() {
            return wordSet;
        }                
    }
    
    // A TreeSet marks num of occurrences 
    class UsefulTreeSet<T> {
        
        Set<T> listOfT = new TreeSet<>();
        Map<T, Integer> itemOccurrences = new HashMap<>();
        int added;
        ThrowsWhenIllegalFeatureException throwsWhenIllegalItemException;
        
        public UsefulTreeSet() {
            this(TextClassifier.this.throwsWhenIllegalItemException);
        }
        
        public UsefulTreeSet(ThrowsWhenIllegalFeatureException exception) {
            this.throwsWhenIllegalItemException = exception;
        }
            
        public void addAdd(Collection<T> items) {
            for(T w: items) {
                if (throwsWhenIllegalItemException != null) {
                    throwsWhenIllegalItemException.when(w.equals(""));                                            
                }
                listOfT.add(w);
                itemOccurrences.put(w, 1 + itemOccurrences.getOrDefault(w, 0));
            }
            added += items.size();
        }
        
        public int getOccurrencesOfItem(T item) {
            return itemOccurrences.getOrDefault(item, 0);
        }
        
        public Set<T> list() {
            return listOfT;
        }
        
        public int getAddedItems() {
            return added;
        }        
    }
    
    interface ThrowsWhenIllegalFeatureException<T> {
        void when(boolean occurs);
    }
}